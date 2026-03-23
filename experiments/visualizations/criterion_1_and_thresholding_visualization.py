#### Criterion 1 and Thresholding Visualization ####
#
# R source: Thresholding_by_Criterion_Visualization/Criterion_1_and_Thresholding_Visualization.R
#
# Uses cropped_img_4.jpg to illustrate how criterion 1 selects the optimal
# intensity threshold from the GP posterior predictive mean.
#
# Pipeline:
#   1. Load image, run GP, extract predmean_mat = predmean[:200, :300]
#   2. Run criterion (delta=0.01, th=0.05×SD of smoothed diff)
#   3. Select 4 example thresholds from the second-difference curve:
#        indx_1 — position of minimum second-diff
#        indx_2 — position of maximum second-diff
#        indx_3 = 20  (hardcoded in R, 1-indexed → 0-indexed: 19)
#        indx_4 = 80  (hardcoded in R, 1-indexed → 0-indexed: 79)
#   4. Show thresholded images for 4 examples + optimal (5 panels)
#   5. |Δ²c*| vs α plot with coloured markers and stability lines
#   6. Pixel-intensity histogram with threshold vlines
#   7. 3D surface with optimal threshold plane
#
# R used plotly for the 3D plot (interactive). Python uses matplotlib 3D (static).
# R had no explicit save calls (interactive session). Python saves 4 PNG files.
#
# Approximation: RGaSP smoothing replaced by gaussian_filter1d(sigma=2).
#
# Path conventions
#   data/cropped_pieces/  — source image
#   results/figures/      — output figures

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
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

FILE_PATH = DATA_DIR / "cropped_img_4.jpg"


# -----------------------------------------------
# compute_criterion_internals
# -----------------------------------------------
def compute_criterion_internals(predmean_mat: np.ndarray, delta: float = 0.01):
    """
    Full criterion 1 computation exposing all intermediate arrays needed
    for the visualization (second_diff, abs_second_diff, stable_index, th, …).

    Approximation: R uses rgasp() for smoothing; Python uses gaussian_filter1d(sigma=2).
    R: sd() uses ddof=1 — matched here.
    """
    percentages = np.round(np.arange(0.0, 1.0 + 1e-12, delta), 10)

    pixel_counts = np.array(
        [threshold_image(predmean_mat, p, count=True) for p in percentages],
        dtype=np.float64,
    )
    diff_pixel_counts = np.abs(np.diff(pixel_counts))          # length = len(percentages) - 1

    # Approximation: gaussian smoothing in place of RGaSP
    smoothed = gaussian_filter1d(diff_pixel_counts, sigma=2.0, mode="nearest")

    # R: second_diff[i] <- diff[i] - diff[i-1]  for i in 2..n  (1-indexed, i=1 is NA)
    # Python: index 0 = NaN, index 1.. = np.diff(smoothed)
    second_diff = np.empty(len(smoothed))
    second_diff[0] = np.nan
    second_diff[1:] = np.diff(smoothed)
    abs_second_diff = np.abs(second_diff)

    max_index = int(np.argmax(smoothed))
    th = 0.05 * float(np.std(smoothed, ddof=1))   # R: 0.05*sd(diff_pixel_counts)

    stable_index = len(smoothed) - 1
    for i in range(max_index + 1, len(smoothed)):
        if abs(smoothed[i] - smoothed[i - 1]) < th:
            stable_index = i
            break

    # R: percentages[stable_index+1]  (1-indexed diff → 1-indexed percentages shift)
    idx = min(stable_index + 1, len(percentages) - 1)
    estimated_percentage = float(percentages[idx])

    return {
        "percentages":          percentages,
        "pixel_counts":         pixel_counts,
        "smoothed":             smoothed,
        "second_diff":          second_diff,
        "abs_second_diff":      abs_second_diff,
        "max_index":            max_index,
        "stable_index":         stable_index,
        "th":                   th,
        "estimated_percentage": estimated_percentage,
    }


# -----------------------------------------------
# main
# -----------------------------------------------
def main() -> None:
    figures_out = RESULTS_DIR / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)

    # ---- Load image ----
    raw = imageio.imread(str(FILE_PATH))
    ori_img = raw[..., 0].astype(np.float64) if raw.ndim == 3 else raw.astype(np.float64)

    # ---- Run GP pipeline ----
    # R: generate_GP_Masks_test(file_path)  — default nugget=F
    print(f"Running GP segmentation on {FILE_PATH.name} ...")
    result   = generate_gp_masks_test(str(FILE_PATH), nugget=False)
    predmean = result.combined_predmean

    # R: predmean_mat <- gp_masks_result$combined_predmean[1:200, 1:300]
    predmean_mat = predmean[:200, :300]

    # ---- Criterion computation ----
    C = compute_criterion_internals(predmean_mat, delta=0.01)
    percentages      = C["percentages"]
    smoothed         = C["smoothed"]
    second_diff      = C["second_diff"]
    abs_second_diff  = C["abs_second_diff"]
    max_index        = C["max_index"]
    stable_index     = C["stable_index"]
    th               = C["th"]
    estimated_pct    = C["estimated_percentage"]

    # x-axis for diff plots: percentages[1:]  (R: percentages[2:length(percentages)])
    x_diff = percentages[1:]   # length = len(smoothed) = len(percentages) - 1

    # ---- 4 selected example indices ----
    # R: indx_3 <- 20;  indx_4 <- 80  (1-indexed positions in second_diff)
    # Python (0-indexed, NaN at 0): position 20 in R → index 19 in Python
    indx_3_py = 19
    indx_4_py = 79

    # R: indx_1 = which(second_diff == min(second_diff, na.rm=T))
    # Python: nanargmin over second_diff (NaN at 0 is skipped)
    indx_1_py = int(np.nanargmin(second_diff))
    indx_2_py = int(np.nanargmax(second_diff))

    # R: pct_X <- percentages[2:length(percentages)][indx_X]
    # = percentages[1:][indx_X - 1] (Python, since R's [indx_X] on 1-indexed subvector)
    # Because second_diff[i] in R (1-indexed, i>=2) maps to second_diff[i-1] in Python,
    # and percentages[2:n][i] = percentages[i] (0-indexed Python):
    pct_1 = float(percentages[indx_1_py])
    pct_2 = float(percentages[indx_2_py])
    pct_3 = float(percentages[indx_3_py])
    pct_4 = float(percentages[indx_4_py])

    # Thresholded images at all 5 points
    thr1 = threshold_image(predmean_mat, pct_1, count=False)
    thr2 = threshold_image(predmean_mat, pct_2, count=False)
    thr3 = threshold_image(predmean_mat, pct_3, count=False)
    thr4 = threshold_image(predmean_mat, pct_4, count=False)
    thr_opt = threshold_image(predmean_mat, estimated_pct, count=False)

    # Pixel intensities for histogram
    intensities = predmean_mat.ravel()
    vmax_pred   = float(predmean_mat.max())

    # ================================================================
    # Figure 1: 5 thresholded images in a 1×5 row
    # R: image2D(thresholded_imageX, col=c("white","grey"))  ×5
    # ================================================================
    fig1, axes = plt.subplots(1, 5, figsize=(15, 3))
    labels = [
        f"min Δ²\n(α={pct_1:.2f})",
        f"max Δ²\n(α={pct_2:.2f})",
        f"idx=20\n(α={pct_3:.2f})",
        f"idx=80\n(α={pct_4:.2f})",
        f"Optimal\n(α={estimated_pct:.2f})",
    ]
    for ax, img, lbl in zip(axes, [thr1, thr2, thr3, thr4, thr_opt], labels):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest", aspect="auto")
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig1.suptitle("Thresholded Images at Selected Criterion Points", fontsize=11, y=1.01)
    fig1.tight_layout()
    p1 = figures_out / "criterion_1_thresholded_examples.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 2: |Δ²c*| vs α  (absolute second difference plot)
    # R: plot(percentages[2:n], abs_second_diff, ...) + coloured points + ablines
    # ================================================================
    # R: ylim=c(0,4000); axis ticks at seq(0,4000,1000) and seq(0,1,0.25)
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    # Main line + points  (NaN at index 0 is skipped automatically)
    ax2.plot(x_diff, abs_second_diff, color="black", linewidth=2, zorder=2)
    ax2.scatter(x_diff, abs_second_diff, s=8, color="black", zorder=2)

    # Coloured marker points
    # R: points(pct_4, abs_second_diff[indx_4], col="orange") — indx_4=80 in R (1-indexed)
    # Python: abs_second_diff[indx_4_py] at x=pct_4
    point_specs = [
        (pct_1, abs_second_diff[indx_1_py], "blue",   "min Δ²"),
        (pct_2, abs_second_diff[indx_2_py], "green",  "max Δ²"),
        (pct_3, abs_second_diff[indx_3_py], "yellow", "idx=20"),
        (pct_4, abs_second_diff[indx_4_py], "orange", "idx=80"),
    ]
    for px, py_val, col, lbl in point_specs:
        ax2.scatter([px], [py_val], color=col, s=60, zorder=3, label=lbl,
                    edgecolors="black", linewidths=0.5)

    # Vertical line: optimal threshold — R: abline(v=percentages[stable_index+1], col="red", lty="dotted")
    ax2.axvline(x=estimated_pct, color="red", linestyle=":", linewidth=2,
                label=f"Optimal α={estimated_pct:.2f}")

    # Horizontal line: th = 0.05*sd — R: abline(h=0.05*sd(diff), col="purple", lty="dotted")
    ax2.axhline(y=th, color="purple", linestyle=":", linewidth=2)
    ax2.text(0.94, th, r"$0.05\,\tau_k$", color="purple", fontsize=12,
             va="bottom", ha="right")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 4000)
    ax2.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax2.set_yticks([0, 1000, 2000, 3000, 4000])
    # R: expression(~alpha[m]) and "|ΔΔc*(α)|"
    ax2.set_xlabel(r"$\alpha_m$", fontsize=13, fontweight="bold")
    ax2.set_ylabel(r"$|\Delta c^*_k(\alpha_m) - \Delta c^*_k(\alpha_{m-1})|$",
                   fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9, frameon=False)
    fig2.tight_layout()
    p2 = figures_out / "criterion_1_second_diff_plot.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 3: Pixel intensity histogram with threshold vlines
    # R: ggplot(df, aes(x=predmean_mat)) + geom_histogram + geom_vline×5 + ...
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    # Histogram — R: geom_histogram(color="black", fill="lightblue")
    ax3.hist(intensities, bins=50, color="lightblue", edgecolor="black")

    # Vertical lines at threshold intensities
    vline_specs = [
        (estimated_pct * vmax_pred, "red",    "solid",  "Optimal Threshold"),
        (pct_1 * vmax_pred,         "blue",   "dashed", None),
        (pct_2 * vmax_pred,         "green",  "dashed", None),
        (pct_3 * vmax_pred,         "yellow", "dashed", None),
        (pct_4 * vmax_pred,         "orange", "dashed", None),
    ]
    for xv, col, ls, lbl in vline_specs:
        ax3.axvline(x=xv, color=col, linestyle=ls, linewidth=0.75, label=lbl)

    # R: annotate("text", x=opt_thresh+0.015, y=9000, label="Optimal Threshold", angle=90, col="red")
    # Approximate: place rotated text near the optimal vline
    opt_x = estimated_pct * vmax_pred
    ylim_top = ax3.get_ylim()[1]
    ax3.text(opt_x + vmax_pred * 0.01, ylim_top * 0.85, "Optimal Threshold",
             rotation=90, color="red", fontsize=10, va="top")

    ax3.set_xlabel("Predictive Mean of Intensity", fontsize=13)
    ax3.set_ylabel("Frequency", fontsize=13)
    ax3.tick_params(labelsize=11)
    fig3.tight_layout()
    p3 = figures_out / "criterion_1_histogram.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {p3.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 4: 3D surface + optimal threshold plane
    # R: plot_ly() |> add_surface(z=height_matrix) |> add_surface(z=threshold_plane, red)
    # Approximation: plotly interactive → matplotlib static 3D
    # ================================================================
    H, W = predmean_mat.shape
    X3d, Y3d = np.meshgrid(np.arange(W), np.arange(H))
    thr_val = estimated_pct * vmax_pred

    fig4 = plt.figure(figsize=(9, 7))
    ax4  = fig4.add_subplot(111, projection="3d")

    ax4.plot_surface(X3d, Y3d, predmean_mat,
                     cmap="viridis", alpha=0.85, linewidth=0, antialiased=False)
    ax4.plot_surface(X3d, Y3d, np.full_like(predmean_mat, thr_val),
                     color="red", alpha=0.2, linewidth=0)
    ax4.text(W, H, thr_val, "Optimal\nThreshold", color="red", fontsize=9)

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_zlabel("Predictive Mean of Intensity", fontsize=9, labelpad=5)
    ax4.tick_params(axis="z", labelsize=8)

    fig4.tight_layout()
    p4 = figures_out / "criterion_1_3d_surface.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {p4.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
