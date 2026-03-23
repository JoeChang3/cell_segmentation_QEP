import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import imageio.v2 as imageio
from PIL import Image

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter1d
from skimage.segmentation import watershed, find_boundaries as sk_find_boundaries
from skimage.measure import label as cc_label
from skimage.morphology import remove_small_objects
import torch
import gpytorch
import qpytorch

# -----------------------------
# 1) Core kernel
# -----------------------------
def matern52_corr(d: np.ndarray, beta: float) -> np.ndarray:
    """
    R: Matern_5_2_funct(d,beta)
    x = sqrt(5)*beta*d
    (1 + x + x^2/3) * exp(-x)
    """
    x = np.sqrt(5.0) * beta * d
    return (1.0 + x + (x * x) / 3.0) * np.exp(-x)


# -----------------------------
# 2) Threshold utilities
# -----------------------------
def threshold_image(mat: np.ndarray, percentage: float, count: bool = True):
    """
    R: threshold_image(mat, percentage, count)
    threshold = percentage * max(mat)
    """
    max_value = np.nanmax(mat)
    thr = percentage * max_value
    mask = (mat > thr).astype(np.uint8)
    return int(mask.sum()) if count else mask


@dataclass
class Criterion1Result:
    thresholded_image: np.ndarray
    pixel_counts: np.ndarray
    diff_pixel_counts: np.ndarray
    estimated_percentage: float


def criterion_1(predmean_mat: np.ndarray, delta: float = 0.01, nugget: bool = True) -> Criterion1Result:
    """
    R: criterion_1(predmean_mat, delta=0.01, nugget=T)

    逻辑：
    - percentages = 0..1 step delta
    - pixel_counts[t] = number of pixels above percentage*max
    - diff_pixel_counts = abs(diff(pixel_counts))
    - smooth diff (R uses rgasp). Python: gaussian smoothing.
    - find max_index, then find first i>max_index where successive diffs become stable
    - if not found -> all background, percentage=1
    """
    percentages = np.round(np.arange(0.0, 1.0 + 1e-12, delta), 10)

    pixel_counts = np.array([threshold_image(predmean_mat, p, count=True) for p in percentages], dtype=np.float64)
    diff_pixel_counts = np.abs(np.diff(pixel_counts))

    # ---- smooth 
    # bandwidth ~ 2*delta in R; here use sigma in index domain
    smoothed = gaussian_filter1d(diff_pixel_counts, sigma=max(1.0, 2.0), mode="nearest")
    diff_sm = smoothed

    max_index = int(np.argmax(diff_sm))
    th = 0.05 * np.std(diff_sm) if np.std(diff_sm) > 0 else 0.0

    stable_index = len(percentages) - 1
    found_stable = False
    for i in range(max_index + 1, len(diff_sm)):
        if abs(diff_sm[i] - diff_sm[i - 1]) < th:
            stable_index = i
            found_stable = True
            break

    # default: all background
    thresholded_image = np.zeros_like(predmean_mat, dtype=np.uint8)
    estimated_percentage = float(percentages[-1])

    if found_stable:
        # R uses percentages[stable_index+1]
        # note: diff_sm is len(percentages)-1
        idx = min(stable_index + 1, len(percentages) - 1)
        estimated_percentage = float(percentages[idx])
        thresholded_image = threshold_image(predmean_mat, estimated_percentage, count=False)

    return Criterion1Result(
        thresholded_image=thresholded_image,
        pixel_counts=pixel_counts,
        diff_pixel_counts=diff_sm,
        estimated_percentage=estimated_percentage,
    )


# -----------------------------
# 3) Post-processing
# -----------------------------
def eliminate_small_areas(gp_masks: np.ndarray, size_threshold: int) -> np.ndarray:
    """
    R: eliminate_small_areas(GP_masks, size_threshold)

    - remove small components for each label unless on boundary.
    - boundary labels: if area < size_threshold/5 -> remove
    """
    mask = gp_masks.copy()
    labels = np.unique(mask[mask > 0])

    nrow, ncol = mask.shape

    for lab in labels:
        lab_mask = (mask == lab)
        area = int(lab_mask.sum())

        on_boundary = (
            lab_mask[0, :].any()
            or lab_mask[-1, :].any()
            or lab_mask[:, 0].any()
            or lab_mask[:, -1].any()
        )

        if (area < size_threshold and (not on_boundary)) or (on_boundary and area < size_threshold / 5.0):
            mask[lab_mask] = 0

    return mask


def find_boundaries_4n(mask: np.ndarray) -> np.ndarray:
    """
    R: find_boundaries(mask) 自写 4邻域边界
    """
    boundary = np.zeros_like(mask, dtype=np.uint8)
    H, W = mask.shape

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if mask[i, j] > 0:
                v = mask[i, j]
                if (mask[i - 1, j] != v) or (mask[i + 1, j] != v) or (mask[i, j - 1] != v) or (mask[i, j + 1] != v):
                    boundary[i, j] = 1
    return boundary


def get_proportion(size: int, target_min: int = 200, target_max: int = 400) -> float:
    """
    R: get_proportion(size, target_min=200, target_max=400)
    divisors: 1/4, 1/3, 1/2, 1
    """
    divisors = [1/4, 1/3, 1/2, 1.0]
    for div in divisors:
        piece = size * div
        if target_min <= piece <= target_max:
            return div
    return 1/4


# -----------------------------
# 4) "Separable GP" smoothing (Practical replacement)
# -----------------------------
@dataclass
class SeparableGPParams:
    beta1: float
    beta2: float
    nu: float


def separable_gp_param_est_stub(img: np.ndarray) -> SeparableGPParams:
   
    H, W = img.shape
    beta1 = 3.0
    beta2 = 3.0
    nu = 1e-3
    return SeparableGPParams(beta1=beta1, beta2=beta2, nu=nu)





class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="matern", nu=2.5):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
        else:
            raise ValueError("kernel_type must be 'rbf' or 'matern'")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def separable_gp_smooth_gpytorch(
    img: np.ndarray,
    kernel_type: str = "matern",
    nu: float = 2.5,
    train_iters: int = 75,
    lr: float = 0.1,
    device: str = "cpu",
    max_points: int = 6000,
) -> np.ndarray:
    """
    use GPyTorch to fit GP，then predict posterior mean on the whole grid
    - img: (H,W) numpy float
    """
    H, W = img.shape
    y = img.astype(np.float64)

    # grid inputs in [0,1]^2
    xs = np.linspace(0, 1, H)
    ys = np.linspace(0, 1, W)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)  # (H*W,2)
    Y = y.ravel()

    # downsample training points if too many
    N = X.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        train_x = torch.from_numpy(X[idx]).float().to(device)
        train_y = torch.from_numpy(Y[idx]).float().to(device)
    else:
        train_x = torch.from_numpy(X).float().to(device)
        train_y = torch.from_numpy(Y).float().to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel_type=kernel_type, nu=nu).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(train_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # predict on full grid
    model.eval()
    likelihood.eval()
    test_x = torch.from_numpy(X).float().to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x)).mean

    predmean = pred.detach().cpu().numpy().reshape(H, W)
    return predmean


def separable_gp_smooth_qepytorch(
    img: np.ndarray,
    kernel_type: str = "matern",
    nu: float = 1.5,
    q_power: float = 2.0,
    train_iters: int = 75,
    lr: float = 0.1,
    device: str = "cpu",
    max_points: int = 6000,
    jitter: float = 1e-4,
) -> np.ndarray:
    """
    用 QePyTorch 拟合 QEP 并预测 mean。
    """
    H, W = img.shape
    y = img.astype(np.float64)

    xs = np.linspace(0, 1, H)
    ys = np.linspace(0, 1, W)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Y = y.ravel()

    N = X.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        train_x = torch.from_numpy(X[idx]).float().to(device)
        train_y = torch.from_numpy(Y[idx]).float().to(device)
    else:
        train_x = torch.from_numpy(X).float().to(device)
        train_y = torch.from_numpy(Y).float().to(device)

    # ---- build model (API names may vary by qpytorch version)
    model = qpytorch.models.ExactQEPModel(
        train_x=train_x,
        train_y=train_y,
        kernel_type=kernel_type,
        q_power=q_power,
        nu=nu,
        jitter=jitter,
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for _ in range(train_iters):
        optimizer.zero_grad()
        out = model(train_x)
        loss = -mll(out, train_y)
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        optimizer.step()

    model.eval()
    test_x = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        pred = model(test_x).mean  # 或者 model.likelihood(model(test_x)).mean 取决于实现
    predmean = pred.detach().cpu().numpy().reshape(H, W)
    return predmean



# -----------------------------
# 5) Main: generate_GP_Masks_test (Python)
# -----------------------------
@dataclass
class GenerateMasksResult:
    ori_images: List[np.ndarray]
    processed_images: List[np.ndarray]
    crit_1_opt_thresholds: List[float]
    connected_parts_count: List[int]
    outliers: List[int]
    combined_predmean: np.ndarray
    combined_thresholded1: np.ndarray
    gp_masks: np.ndarray


def generate_gp_masks_test(
    file_path: str,
    delta: float = 0.01,
    remove_size_threshold: int = 50,
    nugget: bool = True,
) -> GenerateMasksResult:
    """
    R: generate_GP_Masks_test()

    read images -> divide -> first postion estimate -> smooth -> criterion1 -> deal outlier -> get together
    -> distance transform + watershed -> eliminate_small_areas
    """
    img = imageio.imread(file_path)
    if img.ndim == 3:
        img_gray = img[..., 0].astype(np.float64)
    else:
        img_gray = img.astype(np.float64)

    img_height, img_width = img_gray.shape

    row_prop = get_proportion(img_height)
    col_prop = get_proportion(img_width)

    crop_width = int(img_width * col_prop)
    crop_height = int(img_height * row_prop)

    num_pieces_x = max(1, img_width // crop_width)
    num_pieces_y = max(1, img_height // crop_height)

    # adjust crop to fit exactly by integer division
    crop_width = img_width // num_pieces_x
    crop_height = img_height // num_pieces_y

    combined_predmean = np.zeros((img_height, img_width), dtype=np.float64)
    combined_thresholded1 = np.zeros((img_height, img_width), dtype=np.uint8)

    ori_images = []
    processed_images = []
    thresholded1_images = []
    crit_1_opt_thresholds = []
    connected_parts_count = []

    params: Optional[SeparableGPParams] = None
    count = 0

    # ---- 1) process each tile
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = i * crop_width
            y_offset = j * crop_height

            # last piece fill remaining (R has piece_width/piece_height but still crops with crop_width/crop_height)
            piece_width = img_width - x_offset if (i == num_pieces_x - 1) else crop_width
            piece_height = img_height - y_offset if (j == num_pieces_y - 1) else crop_height

            tile = img_gray[y_offset:y_offset + piece_height, x_offset:x_offset + piece_width].copy()

            if params is None:
                params = separable_gp_param_est_stub(tile)

            # predmean = separable_gp_smooth(tile, params)
            predmean = separable_gp_smooth_gpytorch(tile, kernel_type="matern", nu=2.5, train_iters=75, lr=0.1)


            ori_images.append(tile)
            processed_images.append(predmean)

            c1 = criterion_1(predmean, delta=delta, nugget=nugget)
            thr_img = c1.thresholded_image
            thresholded1_images.append(thr_img)
            crit_1_opt_thresholds.append(float(c1.estimated_percentage))

            # connected parts count: R uses bwlabel(thresholded) and length(unique(...))
            cc = cc_label(thr_img > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))

            count += 1

    # ---- 2) outlier detection
    cp = np.array(connected_parts_count, dtype=np.float64)
    mean_cp = float(cp.mean())
    sd_cp = float(cp.std(ddof=0))
    outlier_threshold = 2.0

    if sd_cp == 0:
        outliers = []
    else:
        outliers = list(np.where(np.abs(cp - mean_cp) > outlier_threshold * sd_cp)[0].astype(int))

    # mean threshold excluding outliers
    if len(outliers) < len(crit_1_opt_thresholds):
        keep = [k for k in range(len(crit_1_opt_thresholds)) if k not in outliers]
        mean_thr = float(np.mean([crit_1_opt_thresholds[k] for k in keep]))
    else:
        mean_thr = float(np.mean(crit_1_opt_thresholds))

    # ---- 3) rethreshold outliers
    for idx in outliers:
        predmean = processed_images[idx]
        rethr = threshold_image(predmean, mean_thr, count=False)
        crit_1_opt_thresholds[idx] = mean_thr

        # if > 99% foreground -> revert to background, thr=1 (same as R)
        if rethr.sum() > 0.99 * rethr.size:
            rethr = np.zeros_like(rethr, dtype=np.uint8)
            crit_1_opt_thresholds[idx] = 1.0

        thresholded1_images[idx] = rethr

    # ---- 4) stitch tiles back
    idx = 0
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = i * crop_width
            y_offset = j * crop_height

            predmean = processed_images[idx]
            thr_img = thresholded1_images[idx]

            h, w = predmean.shape
            combined_predmean[y_offset:y_offset + h, x_offset:x_offset + w] = predmean
            combined_thresholded1[y_offset:y_offset + h, x_offset:x_offset + w] = thr_img.astype(np.uint8)

            idx += 1

    # ---- 5) watershed instance segmentation
    # distance map of foreground (equivalent to EBImage::distmap(as.Image(binary)))
    # Inverse distance for watershed often works better, but here mimic typical: watershed on distance map
    dist_map = distance_transform_edt(combined_thresholded1 > 0)
    # Watershed expects markers; EBImage watershed uses its own defaults.
    # We'll use local maxima markers-less variant by using negative distance + binary mask.
    gp_labels = watershed(-dist_map, markers=None, mask=(combined_thresholded1 > 0))
    gp_masks_raw = gp_labels.astype(np.int32)

    # ---- 6) remove small areas
    gp_masks = eliminate_small_areas(gp_masks_raw, remove_size_threshold)

    return GenerateMasksResult(
        ori_images=ori_images,
        processed_images=processed_images,
        crit_1_opt_thresholds=crit_1_opt_thresholds,
        connected_parts_count=connected_parts_count,
        outliers=outliers,
        combined_predmean=combined_predmean,
        combined_thresholded1=combined_thresholded1,
        gp_masks=gp_masks,
    )
