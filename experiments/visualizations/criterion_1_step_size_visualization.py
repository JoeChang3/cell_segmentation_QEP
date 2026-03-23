#### Criterion 1 — Step Size Visualization ####
#
# R source: Thresholding_by_Criterion_Visualization/Criterion_1_Step_Size_Visualization.R
#
# Runs the criterion 1 thresholding algorithm on a cropped image tile four
# times, each with a different stability threshold multiplier:
#   0.05 × SD,  0.1 × SD,  2.5 × SD,  0.01 × SD  (R's original order)
#
# For each multiplier the script shows:
#   • the resulting binary thresholded image
#   • a 3D surface of the GP predictive mean with the optimal threshold plane
#     overlaid in red (R used plotly interactive; Python uses matplotlib 3D)
#
# All R plot calls (image2D, plot_ly/print) were interactive — no save calls.
# Python saves one composite PNG.
#
# Approximation: RGaSP smoothing of diff_pixel_counts is replaced by
# scipy gaussian_filter1d(sigma=2), matching py_core/Modified_Functions_RGasp.py.
#
# Path conventions
#   data/cropped_pieces/   — source image
#   results/figures/       — output figure

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3D projection
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d

from py_core.Modified_Functions_RGasp import (
    generate_gp_masks_test,
    threshold_image,
)


# -----------------------------------------------
# Paths
# -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "cropped_pieces"
RESULTS_DIR  = PROJECT_ROOT / "results"

# R: file_path <- ".../cropped_pieces/cropped_img_4.jpg"
FILE_PATH = DATA_DIR / "cropped_img_4.jpg"


# -----------------------------------------------
# run_criterion_with_th
# -----------------------------------------------
def run_criterion_with_th(predmean_mat: np.ndarray, th_multiplier: float, delta: float = 0.01):
    """
    Inline criterion 1 logic allowing an arbitrary stability-threshold multiplier.
    R: th <- multiplier * sd(diff_pixel_counts) after RGaSP smoothing.
    Python: uses gaussian_filter1d(sigma=2) in place of rgasp().

    Returns dict with keys needed for visualization.
    """
    percentages = np.round(np.arange(0.0, 1.0 + 1e-12, delta), 10)

    pixel_counts = np.array(
        [threshold_image(predmean_mat, p, count=True) for p in percentages],
        dtype=np.float64,
    )
    diff_pixel_counts = np.abs(np.diff(pixel_counts))

    # Approximation: R uses rgasp() 1-D GP smoother; Python uses Gaussian smoothing
    smoothed = gaussian_filter1d(diff_pixel_counts, sigma=2.0, mode="nearest")

    max_index = int(np.argmax(smoothed))
    th = th_multiplier * float(np.std(smoothed, ddof=1))   # R: sd() uses ddof=1

    stable_index = len(smoothed) - 1   # default: last index
    for i in range(max_index + 1, len(smoothed)):
        if abs(smoothed[i] - smoothed[i - 1]) < th:
            stable_index = i
            break

    # R: estimated_percentage <- percentages[stable_index + 1]  (1-indexed)
    idx = min(stable_index + 1, len(percentages) - 1)
    estimated_percentage = float(percentages[idx])

    thresholded = threshold_image(predmean_mat, estimated_percentage, count=False)

    return {
        "thresholded_image":      thresholded,
        "estimated_percentage":   estimated_percentage,
        "threshold_value":        estimated_percentage * float(predmean_mat.max()),
        "smoothed":               smoothed,
        "percentages":            percentages,
        "max_index":              max_index,
        "stable_index":           stable_index,
        "th":                     th,
    }


# -----------------------------------------------
# main
# -----------------------------------------------
def main() -> None:
    figures_out = RESULTS_DIR / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)

    # ---- Load image ----
    # R: ori_img_matrix <- as.numeric(img[[1]])[,,1]
    raw = imageio.imread(str(FILE_PATH))
    ori_img = raw[..., 0].astype(np.float64) if raw.ndim == 3 else raw.astype(np.float64)

    # ---- Run GP pipeline ----
    # R: generate_GP_Masks_test(file_path)  — default nugget=F
    print(f"Running GP segmentation on {FILE_PATH.name} ...")
    result   = generate_gp_masks_test(str(FILE_PATH), nugget=False)
    predmean = result.combined_predmean

    # R: predmean_mat <- gp_masks_result$combined_predmean[1:200, 1:300]
    # R is 1-indexed inclusive; Python: [:200, :300]
    predmean_mat = predmean[:200, :300]

    # ---- Four th multipliers in R's original order ----
    # R sections: "0.05*SD", "0.1*SD", "2.5*SD", "0.01*SD"
    th_multipliers = [0.05, 0.1, 2.5, 0.01]

    # ---- Build composite figure: 4 rows × 2 cols ----
    # col 0: binary thresholded image  |  col 1: 3D surface + threshold plane
    fig = plt.figure(figsize=(12, 16))

    nrows = len(th_multipliers)
    row_label_xs = 0.01
    fig.text(0.28, 0.97, "Thresholded Image",    ha="center", fontsize=11, fontweight="bold")
    fig.text(0.72, 0.97, "GP Surface + Threshold Plane", ha="center", fontsize=11, fontweight="bold")

    # Precompute meshgrid for 3D surface once (same predmean_mat for all)
    H, W = predmean_mat.shape
    col_coords = np.arange(W)
    row_coords = np.arange(H)
    X3d, Y3d = np.meshgrid(col_coords, row_coords)   # X=cols, Y=rows

    for row_idx, th_mult in enumerate(th_multipliers):
        res = run_criterion_with_th(predmean_mat, th_multiplier=th_mult)
        thr_img  = res["thresholded_image"]
        thr_val  = res["threshold_value"]
        pct      = res["estimated_percentage"]

        row_label = f"δ = {th_mult}×SD\n(opt. thr = {pct:.3f})"

        # ---- Col 0: binary thresholded image ----
        # R: image2D(thresholded_image, col=c("white","grey"), ...)
        ax_img = fig.add_subplot(nrows, 2, row_idx * 2 + 1)
        ax_img.imshow(thr_img, cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest", aspect="auto")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_ylabel(row_label, fontsize=9, labelpad=4)

        # ---- Col 1: 3D surface + threshold plane ----
        # R: plot_ly() |> add_surface(z=height_matrix) |> add_surface(z=threshold_plane, red)
        # Approximation: plotly interactive 3D → matplotlib static 3D surface.
        ax3d = fig.add_subplot(nrows, 2, row_idx * 2 + 2, projection="3d")

        # Main GP surface
        ax3d.plot_surface(X3d, Y3d, predmean_mat,
                          cmap="viridis", alpha=0.85, linewidth=0, antialiased=False)

        # Threshold plane (constant z = thr_val) — R: col=c("red","red"), opacity=0.2
        ax3d.plot_surface(X3d, Y3d,
                          np.full_like(predmean_mat, thr_val),
                          color="red", alpha=0.2, linewidth=0)

        # Annotation label — R: add_annotations(text="Optimal Threshold", ...)
        ax3d.text(W, H, thr_val, "Optimal\nThreshold", color="red", fontsize=7)

        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zlabel("Pred. Mean Intensity", fontsize=7, labelpad=2)
        ax3d.tick_params(axis="z", labelsize=6)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.02,
                        hspace=0.25, wspace=0.15)

    out_path = figures_out / "criterion_1_step_size_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
