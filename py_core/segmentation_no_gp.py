# Segmentation pipeline without GP smoothing.
#
# R source: Nuclear_Real_Analysis/GP_vs_NoGP/
#           Segmentation_Functions_NoGP (no conflict function names).R
#
# Functions whose R counterparts ("2"-suffix variants) are *identical* to the
# base versions are NOT redefined here — they are simply imported from
# Modified_Functions_RGasp.py:
#   threshold_image2  ->  threshold_image
#   get_proportion2   ->  get_proportion
#   find_boundaries2  ->  find_boundaries_4n
#
# New functions in this module (genuinely different from the base GP versions):
#   criterion_1_no_smooth          (criterion_1_2)
#   eliminate_small_areas_relative (eliminate_small_areas2)
#   generate_no_gp_masks           (generate_GP_Masks2)
#   generate_no_gp_masks_test      (generate_GP_Masks_test2)

import math
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label as cc_label

from py_core.Modified_Functions_RGasp import (
    threshold_image,
    get_proportion,
    find_boundaries_4n,
    GenerateMasksResult,
)


# -----------------------------
# criterion_1_no_smooth
# -----------------------------
def criterion_1_no_smooth(predmean_mat: np.ndarray, delta: float = 0.01, nugget: bool = True) -> dict:
    """
    R: criterion_1_2(predmean_mat, delta=0.01, nugget=T)

    Identical logic to criterion_1 in Modified_Functions_RGasp.py EXCEPT that
    no smoothing is applied to diff_pixel_counts.  The R file contains several
    commented-out smoothing options (ksmooth, rgasp) and uses the raw
    diff_pixel_counts directly.

    Approximation: R's sd() uses ddof=1 (sample std).  np.std(..., ddof=1)
    matches this.  If diff_pixel_counts has length 1, std=NaN; th is set to
    0.0 in that case and no stable index will be found.
    """
    percentages = np.round(np.arange(0.0, 1.0 + 1e-12, delta), 10)

    pixel_counts = np.array(
        [threshold_image(predmean_mat, p, count=True) for p in percentages],
        dtype=np.float64,
    )
    diff_pixel_counts = np.abs(np.diff(pixel_counts))  # length = len(percentages) - 1

    max_index = int(np.argmax(diff_pixel_counts))
    # R: th <- 0.05 * sd(diff_pixel_counts)  [ddof=1, sample std]
    th = 0.05 * float(np.std(diff_pixel_counts, ddof=1)) if len(diff_pixel_counts) > 1 else 0.0

    stable_index = len(percentages) - 1
    found_stable = False
    for i in range(max_index + 1, len(diff_pixel_counts)):
        if abs(diff_pixel_counts[i] - diff_pixel_counts[i - 1]) < th:
            stable_index = i
            found_stable = True
            break

    # default: all background
    thresholded_image = np.zeros(predmean_mat.shape, dtype=np.uint8)
    estimated_percentage = float(percentages[-1])

    if found_stable:
        # R: percentages[stable_index + 1]  (1-based, so offset by +1)
        # stable_index is a 0-based index into diff_pixel_counts (length N-1).
        # percentages[stable_index + 1] moves one step forward in the percentages array.
        idx = min(stable_index + 1, len(percentages) - 1)
        estimated_percentage = float(percentages[idx])
        thresholded_image = threshold_image(predmean_mat, estimated_percentage, count=False)

    return {
        "thresholded_image": thresholded_image,
        "pixel_counts": pixel_counts,
        "diff_pixel_counts": diff_pixel_counts,
        "estimated_percentage": estimated_percentage,
    }


# -----------------------------
# eliminate_small_areas_relative
# -----------------------------
def eliminate_small_areas_relative(gp_masks: np.ndarray) -> np.ndarray:
    """
    R: eliminate_small_areas2(GP_masks)

    Unlike eliminate_small_areas() in Modified_Functions_RGasp.py (which takes a
    fixed size_threshold parameter), this version computes the mean object area
    from the mask itself and uses relative thresholds:
      - interior objects with area < 0.15 * mean_obj_size are removed
      - boundary objects with area < 0.05 * mean_obj_size are removed

    If the mask contains no non-zero labels, the original mask is returned
    unchanged.
    """
    mask = gp_masks.copy()
    nonzero_pixels = mask[mask > 0]

    if nonzero_pixels.size == 0:
        return mask

    # R: label_counts <- table(as.vector(GP_masks))[names != 0]
    # Each entry is the pixel count for that label.
    labels, counts = np.unique(nonzero_pixels, return_counts=True)
    mean_obj_size = float(counts.mean())

    nrow, ncol = mask.shape

    for lab, area in zip(labels, counts):
        lab_mask = mask == lab
        on_boundary = (
            lab_mask[0, :].any()
            or lab_mask[-1, :].any()
            or lab_mask[:, 0].any()
            or lab_mask[:, -1].any()
        )

        if not on_boundary and area < mean_obj_size * 0.15:
            mask[lab_mask] = 0
        elif on_boundary and area < mean_obj_size * 0.05:
            mask[lab_mask] = 0

    return mask


# -----------------------------
# _load_image_gray
# -----------------------------
def _load_image_gray(file_path: str) -> np.ndarray:
    """Load an image and return a 2-D float64 array (first channel)."""
    img = imageio.imread(file_path)
    if img.ndim == 3:
        return img[..., 0].astype(np.float64)
    return img.astype(np.float64)


# -----------------------------
# _run_watershed
# -----------------------------
def _run_watershed(combined_thresholded1: np.ndarray) -> np.ndarray:
    """
    R: distmap(as.Image(combined_thresholded1)); EBImage::watershed(dist_map)
    Approximation: scipy distance_transform_edt + skimage watershed on -dist_map.
    EBImage's watershed uses its own internal marker detection; skimage without
    explicit markers uses a markerless variant that may produce slightly
    different label counts on borderline cases.
    """
    binary = (combined_thresholded1 > 0)
    dist_map = distance_transform_edt(binary)
    labels = watershed(-dist_map, markers=None, mask=binary)
    return labels.astype(np.int32)


# -----------------------------
# _detect_outliers
# -----------------------------
def _detect_outliers(connected_parts_count: list) -> list:
    """
    R: outliers <- which(abs(cp - mean_cp) > 2 * sd_cp)
    R's sd() uses ddof=1.  Returns 0-based indices into the list.
    Approximation: if all counts are equal (sd=0) no outliers are flagged,
    matching R's behaviour (sd=0 means |cp - mean| = 0 which is never > 0).
    """
    cp = np.array(connected_parts_count, dtype=np.float64)
    if len(cp) < 2:
        return []
    sd_cp = float(np.std(cp, ddof=1))
    if sd_cp == 0.0:
        return []
    mean_cp = float(cp.mean())
    return list(np.where(np.abs(cp - mean_cp) > 2.0 * sd_cp)[0].astype(int))


# -----------------------------
# generate_no_gp_masks
# -----------------------------
def generate_no_gp_masks(
    file_path: str,
    delta: float = 0.01,
    nugget: bool = True,
) -> GenerateMasksResult:
    """
    R: generate_GP_Masks2(file_path, delta=0.01, nugget=T)

    The "NoGP" pipeline.  Key difference from generate_gp_masks_test():
      - No GP model is fitted.  predmean_mat = raw image tile (img_matrix).
      - Criterion 1 is applied with no smoothing (criterion_1_no_smooth).
      - Small-area removal uses relative mean-based thresholds.
      - Tile count uses ceiling(1 / proportion) rather than floor(size / crop).

    R uses image_read (magick), which normalises pixel values to [0, 1].
    Python uses imageio, which returns uint8 [0, 255].  Because criterion_1
    thresholds as a fraction of max(mat), the absolute scale does not affect
    the result.
    """
    img_gray = _load_image_gray(file_path)
    img_height, img_width = img_gray.shape

    row_prop = get_proportion(img_height)
    col_prop = get_proportion(img_width)

    crop_width = img_width * col_prop      # R keeps float here
    crop_height = img_height * row_prop

    # R: ceiling(1 / col_proportion)
    num_pieces_x = math.ceil(1.0 / col_prop)
    num_pieces_y = math.ceil(1.0 / row_prop)

    ori_images: list = []
    processed_images: list = []
    thresholded1_images: list = []
    crit_1_opt_thresholds: list = []
    connected_parts_count: list = []

    # ---- 1) process each tile
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = int(round(i * crop_width))
            y_offset = int(round(j * crop_height))
            pw = int(round(crop_width))
            ph = int(round(crop_height))

            tile = img_gray[y_offset:y_offset + ph, x_offset:x_offset + pw].copy()

            # NoGP: predmean_mat = raw image (no GP fitting)
            predmean_mat = tile

            ori_images.append(tile)
            processed_images.append(predmean_mat)

            c1 = criterion_1_no_smooth(predmean_mat, delta=delta, nugget=nugget)
            thr_img = c1["thresholded_image"]
            thresholded1_images.append(thr_img)
            crit_1_opt_thresholds.append(float(c1["estimated_percentage"]))

            # R: length(unique(as.vector(bwlabel(thresholded1_img))))
            cc = cc_label(thr_img > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))

    # ---- 2) outlier detection
    outliers = _detect_outliers(connected_parts_count)

    keep = [k for k in range(len(crit_1_opt_thresholds)) if k not in outliers]
    mean_thr = float(np.mean([crit_1_opt_thresholds[k] for k in keep])) if keep else float(np.mean(crit_1_opt_thresholds))

    # ---- 3) rethreshold outliers (generate_GP_Masks2 has no 99% guard)
    for idx in outliers:
        rethr = threshold_image(processed_images[idx], mean_thr, count=False)
        crit_1_opt_thresholds[idx] = mean_thr
        thresholded1_images[idx] = rethr

    # ---- 4) stitch tiles
    combined_predmean = np.zeros((img_height, img_width), dtype=np.float64)
    combined_thresholded1 = np.zeros((img_height, img_width), dtype=np.uint8)

    tile_idx = 0
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = int(round(i * crop_width))
            y_offset = int(round(j * crop_height))
            pm = processed_images[tile_idx]
            ti = thresholded1_images[tile_idx]
            h, w = pm.shape
            combined_predmean[y_offset:y_offset + h, x_offset:x_offset + w] = pm
            combined_thresholded1[y_offset:y_offset + h, x_offset:x_offset + w] = ti.astype(np.uint8)
            tile_idx += 1

    # ---- 5) watershed + small-area removal
    gp_masks_raw = _run_watershed(combined_thresholded1)
    gp_masks = eliminate_small_areas_relative(gp_masks_raw)

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


# -----------------------------
# generate_no_gp_masks_test
# -----------------------------
def generate_no_gp_masks_test(
    file_path: str,
    delta: float = 0.01,
    nugget: bool = True,
) -> GenerateMasksResult:
    """
    R: generate_GP_Masks_test2(file_path, delta=0.01, nugget=T)

    The practical NoGP pipeline used in real-data analysis.  Differences from
    generate_no_gp_masks():
      - Tile count uses floor(size / crop) so tiles divide the image exactly
        (integer arithmetic, no fractional offsets).  The last tile absorbs
        any remainder pixels.
      - Outlier rethresholding includes a "≥99% foreground" guard: if the
        rethresholded tile has more than 99% foreground pixels, it is reverted
        to all-background and its threshold is set to 1.
    """
    img_gray = _load_image_gray(file_path)
    img_height, img_width = img_gray.shape

    row_prop = get_proportion(img_height)
    col_prop = get_proportion(img_width)

    # R: crop_width <- as.integer(img_width * col_proportion)
    crop_width = int(img_width * col_prop)
    crop_height = int(img_height * row_prop)

    # R: num_pieces_x <- floor(img_width / crop_width)
    num_pieces_x = max(1, img_width // crop_width)
    num_pieces_y = max(1, img_height // crop_height)

    # R: crop_width <- img_width %/% num_pieces_x  (integer division re-normalise)
    crop_width = img_width // num_pieces_x
    crop_height = img_height // num_pieces_y

    ori_images: list = []
    processed_images: list = []
    thresholded1_images: list = []
    crit_1_opt_thresholds: list = []
    connected_parts_count: list = []

    # ---- 1) process each tile
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = i * crop_width
            y_offset = j * crop_height

            # last piece absorbs remainder pixels
            piece_width = (img_width - x_offset) if (i == num_pieces_x - 1) else crop_width
            piece_height = (img_height - y_offset) if (j == num_pieces_y - 1) else crop_height

            tile = img_gray[y_offset:y_offset + piece_height, x_offset:x_offset + piece_width].copy()

            # NoGP: predmean_mat = raw image (no GP fitting)
            predmean_mat = tile

            ori_images.append(tile)
            processed_images.append(predmean_mat)

            c1 = criterion_1_no_smooth(predmean_mat, delta=delta, nugget=nugget)
            thr_img = c1["thresholded_image"]
            thresholded1_images.append(thr_img)
            crit_1_opt_thresholds.append(float(c1["estimated_percentage"]))

            cc = cc_label(thr_img > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))

    # ---- 2) outlier detection
    outliers = _detect_outliers(connected_parts_count)

    keep = [k for k in range(len(crit_1_opt_thresholds)) if k not in outliers]
    mean_thr = float(np.mean([crit_1_opt_thresholds[k] for k in keep])) if keep else float(np.mean(crit_1_opt_thresholds))

    # ---- 3) rethreshold outliers (with 99% foreground guard)
    for idx in outliers:
        rethr = threshold_image(processed_images[idx], mean_thr, count=False)
        crit_1_opt_thresholds[idx] = mean_thr

        # R: if(sum > 0.99 * nrow * ncol) revert to background, thr = 1
        if rethr.sum() > 0.99 * rethr.size:
            rethr = np.zeros_like(rethr, dtype=np.uint8)
            crit_1_opt_thresholds[idx] = 1.0

        thresholded1_images[idx] = rethr

    # ---- 4) stitch tiles
    combined_predmean = np.zeros((img_height, img_width), dtype=np.float64)
    combined_thresholded1 = np.zeros((img_height, img_width), dtype=np.uint8)

    tile_idx = 0
    for i in range(num_pieces_x):
        for j in range(num_pieces_y):
            x_offset = i * crop_width
            y_offset = j * crop_height
            pm = processed_images[tile_idx]
            ti = thresholded1_images[tile_idx]
            h, w = pm.shape
            combined_predmean[y_offset:y_offset + h, x_offset:x_offset + w] = pm
            combined_thresholded1[y_offset:y_offset + h, x_offset:x_offset + w] = ti.astype(np.uint8)
            tile_idx += 1

    # ---- 5) watershed + small-area removal
    gp_masks_raw = _run_watershed(combined_thresholded1)
    gp_masks = eliminate_small_areas_relative(gp_masks_raw)

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
