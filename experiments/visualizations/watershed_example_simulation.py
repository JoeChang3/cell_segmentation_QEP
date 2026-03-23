#### Watershed Example Simulation ####
#
# R source: Watershed_Simulation/Watershed_Example_Simulation.R
#
# Demonstrates the watershed segmentation algorithm on a two-cell crop from a
# whole-image GP segmentation result.  The script:
#
#   1. Loads cropped_img_2.jpg, runs GP, extracts the full GP mask.
#   2. Crops to a two-cell region: GP_masks[344:412, 694:735]  (R 1-indexed).
#      Python 0-indexed slice: [343:412, 693:735]
#   3. Builds binary mask, Euclidean distance map, negative distance map.
#   4. Loads a pre-prepared "diagonal" path matrix from data/different_size_diag_cropping_2.csv.
#      (Originally prepared in Excel as noted in R comments.)
#   5. Extracts heights along the diagonal, finds local minima per cell label.
#   6. Produces watershed ribbon plots at three water levels (-13, -10.20, 0).
#   7. Runs watershed on the two-cell binary and shows the coloured result.
#   8. Produces overlay plots of the cropped original image with per-cell colour masks.
#
# R notes:
#   • R's image2D() shows matrices with row→x, col→y (transposed vs Python imshow).
#     Python uses standard imshow orientation (row→y, col→x).
#     All scatter overlays use the correct (x=col, y=row) convention accordingly.
#   • R's watershed(dist_map) comes from EBImage.  Python uses
#     skimage.segmentation.watershed(-dist_map, mask=binary>0).
#   • R had no explicit ggsave() calls; all plots were shown interactively.
#     Python saves multiple PNG files to results/figures/.
#   • dist_map and watershed_result are the size of the cropped region only.
#     Overlay points check neg_dist <= level within the crop.
#   • watershed_height = -10.198039 and watershed_position = 14 are hardcoded
#     from R's interactive exploratory analysis and preserved here.
#
# Path conventions
#   data/cropped_pieces/                       — source image
#   data/different_size_diag_cropping_2.csv    — diagonal path matrix
#   results/figures/                           — output figures

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


# -----------------------------------------------
# Paths
# -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

FILE_PATH   = DATA_DIR / "cropped_pieces" / "cropped_img_2.jpg"
DIAG_CSV    = DATA_DIR / "different_size_diag_cropping_2.csv"


# -----------------------------------------------
# add_water
# -----------------------------------------------
def add_water(heights: np.ndarray, water_level: float) -> np.ndarray:
    """
    R: add_water <- function(df, water_level) { df$water <- pmax(water_level, df$height) }
    Returns the water surface y-values at each position.
    """
    return np.maximum(water_level, heights)


# -----------------------------------------------
# generate_watershed_plot
# -----------------------------------------------
def generate_watershed_plot(ax, x: np.ndarray, heights: np.ndarray, water_level: float) -> None:
    """
    R: generate_watershed_plot(water_level)
    ggplot ribbon (geom_ribbon) between height and water + landscape line (geom_line) +
    horizontal dashed water-level line.
    Python: fill_between + plot + axhline.
    """
    water = add_water(heights, water_level)
    ax.fill_between(x, heights, water, color="lightblue", alpha=0.7)    # geom_ribbon
    ax.plot(x, heights, color="brown", linewidth=1)                      # geom_line
    ax.axhline(y=water_level, color="blue", linestyle="--")              # geom_hline
    ax.set_ylim(-16, 1)
    ax.tick_params(labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


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
    # Import here to avoid circular imports at module level
    from py_core.Modified_Functions_RGasp import generate_gp_masks_test
    print(f"Running GP segmentation on {FILE_PATH.name} ...")
    result   = generate_gp_masks_test(str(FILE_PATH), nugget=False)
    gp_masks = result.gp_masks

    # ---- Crop to two-cell region ----
    # R: gp_mask_select <- GP_masks[344:412, 694:735]  (1-indexed inclusive)
    # Python 0-indexed: [343:412, 693:735]
    gp_mask_select = gp_masks[343:412, 693:735].copy()
    ori_crop       = ori_img[343:412, 693:735].copy()

    # ---- Binary mask ----
    # R: bin_two[which(gp_mask_select > 0)] <- 1
    bin_two = (gp_mask_select > 0).astype(np.float32)

    # ---- Distance map ----
    # R: dist_map <- distmap(bin_two)  (EBImage::distmap = Euclidean distance transform)
    # Approximation: scipy distance_transform_edt matches for binary inputs.
    dist_map = distance_transform_edt(bin_two > 0)
    neg_dist = -dist_map

    # ---- Load diagonal CSV ----
    # R: modified_neg_dist <- as.matrix(read.csv(path))[,-1]  (drops row-name column)
    modified_neg_dist = pd.read_csv(str(DIAG_CSV), index_col=0).values.astype(np.float64)

    # ---- Diagonal path indices ----
    # R: indices <- which(modified_neg_dist == 1, arr.ind=TRUE)  (1-indexed row/col)
    # Python: argwhere gives 0-indexed [row, col]
    indices = np.argwhere(modified_neg_dist == 1)   # shape (n_pts, 2)

    # Heights along the diagonal
    # R: heights <- neg_dist[indices]  (selects elements by [row,col] pairs, 1-indexed)
    heights_diag = neg_dist[indices[:, 0], indices[:, 1]]

    # ---- Find local minima per cell label (for initial landscape plot) ----
    # R: switch_vis <- cbind(neg_dist[indices], gp_mask_select[indices])
    #    df_switch groups by label, finds min_height per label
    labels_diag = gp_mask_select[indices[:, 0], indices[:, 1]]
    switch_arr  = np.column_stack([heights_diag, labels_diag])   # (n, 2)

    unique_labels = np.unique(labels_diag)  # includes 0 (background) if present
    df_switch = []
    for lbl in unique_labels:
        mask_lbl = labels_diag == lbl
        df_switch.append({"label": lbl, "min_height": heights_diag[mask_lbl].min()})

    # R: df_switch$min_height[2] and [3] refer to the 2nd and 3rd rows (1-indexed)
    # which are the first two non-background labels (assuming sorted unique labels)
    df_switch = sorted(df_switch, key=lambda d: d["label"])
    # Take the 2nd and 3rd entries (R 1-indexed → Python [1] and [2])
    cell_a = df_switch[1] if len(df_switch) > 1 else df_switch[0]
    cell_b = df_switch[2] if len(df_switch) > 2 else df_switch[1]

    # Position and height of each cell's minimum along the diagonal
    pos_a = int(np.where((heights_diag == cell_a["min_height"]) &
                         (labels_diag  == cell_a["label"]))[0][0]) + 1   # 1-based for plot
    pos_b = int(np.where((heights_diag == cell_b["min_height"]) &
                         (labels_diag  == cell_b["label"]))[0][0]) + 1

    x_diag = np.arange(1, len(heights_diag) + 1, dtype=float)   # 1-based positions (R: 1:n)

    # ---- Hardcoded watershed line (from R's exploratory analysis) ----
    # R: watershed_height <- -10.198039; watershed_position <- 14
    WATERSHED_HEIGHT   = -10.198039
    WATERSHED_POSITION = 14

    # ---- Watershed segmentation ----
    # R: watershed_result <- watershed(dist_map)  (EBImage)
    # Approximation: skimage watershed with negative distance map and no explicit markers.
    watershed_result = watershed(-dist_map, mask=bin_two > 0)

    # ================================================================
    # Figure 1: Overview — binary, dist, neg_dist, neg_dist with diagonal marked
    # ================================================================
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    titles1 = ["Binary Mask", "Distance Map", "Neg. Distance Map", "Neg. Dist. + Diagonal"]

    # R: image2D(bin_two, col=c("white","grey"), ...)
    axes1[0].imshow(bin_two.T, cmap="gray_r", vmin=0, vmax=1,
                    interpolation="nearest", aspect="auto")

    # R: image2D(dist_map, ...)
    axes1[1].imshow(dist_map.T, cmap="viridis", interpolation="nearest", aspect="auto")

    # R: image2D(neg_dist, ...)
    axes1[2].imshow(neg_dist.T, cmap="viridis_r", interpolation="nearest", aspect="auto")

    # R: neg_dist_diag <- neg_dist; neg_dist_diag[indices] <- 1; image2D(neg_dist_diag)
    neg_dist_diag = neg_dist.copy()
    neg_dist_diag[indices[:, 0], indices[:, 1]] = 1.0
    axes1[3].imshow(neg_dist_diag.T, cmap="viridis", interpolation="nearest", aspect="auto")

    for ax, t in zip(axes1, titles1):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.tight_layout()
    p1 = figures_out / "watershed_simulation_overview.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 2: Heights along diagonal
    # R: plot(heights, xlab="Position", ylab="Height")
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(x_diag, heights_diag, "o", markersize=4, color="black")
    ax2.set_xlabel("Position", fontsize=12)
    ax2.set_ylabel("Height", fontsize=12)
    ax2.set_title("Heights Along Diagonal Path", fontsize=11)
    fig2.tight_layout()
    p2 = figures_out / "watershed_simulation_diagonal_heights.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 3: Watershed landscape ribbon plots (4 panels)
    #   Panel 0: Initial — landscape only with local minima (no water)
    #   Panel 1: water_level = -13   (R: water_levels_demo <- c(-13))
    #   Panel 2: water_level = watershed_height = -10.198039 + watershed vline
    #   Panel 3: water_level = 0 (full) + watershed vline
    # ================================================================
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    panel_titles3 = [
        "Initial (local minima)",
        "Water level = -13",
        f"Water level ≈ {WATERSHED_HEIGHT:.2f}\n(watershed forms)",
        "Water level = 0 (full)",
    ]

    # Panel 0: landscape only + local minima (purple and orange points)
    # R: ggplot + geom_line + geom_point(col=c("purple","orange"))
    axes3[0].plot(x_diag, heights_diag, color="brown", linewidth=1)
    axes3[0].set_ylim(-16, 1)
    axes3[0].scatter([pos_a], [cell_a["min_height"]], color="purple", s=60, zorder=3)
    axes3[0].scatter([pos_b], [cell_b["min_height"]], color="orange", s=60, zorder=3)
    axes3[0].tick_params(labelsize=11)

    # Panel 1: water at -13
    generate_watershed_plot(axes3[1], x_diag, heights_diag, water_level=-13.0)

    # Panel 2: water at watershed_height + vertical watershed line
    # R: final_plot <- generate_watershed_plot(round(watershed_height,2)) +
    #                  geom_vline(xintercept=watershed_position, col="red", lty="dashed")
    generate_watershed_plot(axes3[2], x_diag, heights_diag, water_level=round(WATERSHED_HEIGHT, 2))
    axes3[2].axvline(x=WATERSHED_POSITION, color="red", linestyle="--")

    # Panel 3: water at 0 (full) + vertical watershed line
    generate_watershed_plot(axes3[3], x_diag, heights_diag, water_level=0.0)
    axes3[3].axvline(x=WATERSHED_POSITION, color="red", linestyle="--")

    for ax, t in zip(axes3, panel_titles3):
        ax.set_title(t, fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig3.tight_layout()
    p3 = figures_out / "watershed_simulation_landscape_plots.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {p3.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 4: Watershed result
    # R: image2D(watershed_result, col=hcl.colors(100,"plasma"), ...)
    # Approximation: matplotlib "plasma" colormap
    # ================================================================
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    ax4.imshow(watershed_result.T, cmap="plasma", interpolation="nearest", aspect="auto")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title("Watershed Result", fontsize=11)
    fig4.tight_layout()
    p4 = figures_out / "watershed_simulation_result.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {p4.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 5: Original image overlays at 3 water levels
    # Three sub-panels per water level: base image + cell-A (purple) + cell-B (orange)
    # R: three separate image2D calls with points() overlay per level
    # ================================================================
    all_levels = [-13.0, WATERSHED_HEIGHT, 0.0]
    level_labels = ["Water = -13", f"Water ≈ {WATERSHED_HEIGHT:.2f}", "Water = 0"]

    # Infer cell labels from watershed_result (labels 1 and 2)
    ws_labels = [l for l in np.unique(watershed_result) if l > 0]
    label_a = ws_labels[0] if len(ws_labels) >= 1 else 1
    label_b = ws_labels[1] if len(ws_labels) >= 2 else 2

    # Grayscale values for the original crop  (R: grey.colors(n_unique_vals))
    n_grey = len(np.unique(ori_crop.astype(int)))
    grey_cmap = plt.cm.gray

    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
    for ax5, level, lbl in zip(axes5, all_levels, level_labels):
        ax5.imshow(ori_crop.T, cmap=grey_cmap, interpolation="nearest", aspect="auto")

        # Purple cell (watershed label_a): neg_dist <= level within label region
        # R: which(neg_dist <= level & watershed_result == 1, arr.ind=T)
        mask_a = (neg_dist <= level) & (watershed_result == label_a)
        rows_a, cols_a = np.where(mask_a)
        if len(rows_a):
            # R: points(row/nrow, col/ncol) on image2D → scatter at (col, row) on imshow
            ax5.scatter(rows_a, cols_a,
                        color=(0.5, 0, 0.5, 0.3), s=80, marker="o", linewidths=0)

        # Orange cell (watershed label_b): neg_dist < level (R uses strict <)
        # R: which(neg_dist < level & watershed_result == 2, arr.ind=T)
        mask_b = (neg_dist < level) & (watershed_result == label_b)
        rows_b, cols_b = np.where(mask_b)
        if len(rows_b):
            ax5.scatter(rows_b, cols_b,
                        color=(1.0, 0.647, 0, 0.3), s=80, marker="o", linewidths=0)

        ax5.set_title(lbl, fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])

    fig5.suptitle("Original Image with Water-Level Cell Overlays", fontsize=12)
    fig5.tight_layout()
    p5 = figures_out / "watershed_simulation_overlay_plots.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved: {p5.relative_to(PROJECT_ROOT)}")

    # ================================================================
    # Figure 6: Original crop — with and without diagonal
    # R: image2D(ori_with_diagonal) + lines(indices/nrow, indices/ncol, col="red")
    # ================================================================
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(10, 5))
    H_c, W_c = ori_crop.shape

    # Without diagonal
    ax6a.imshow(ori_crop.T, cmap="gray", interpolation="nearest", aspect="auto")
    ax6a.set_title("Without Diagonal", fontsize=11)
    ax6a.set_xticks([])
    ax6a.set_yticks([])

    # With diagonal
    # R: lines(indices[,1]/nrow, indices[,2]/ncol, col="red", lwd=2)
    # image2D has row→x, col→y. Python imshow has col→x, row→y.
    # For the diagonal overlay: rows_d / H_c → x on image2D axes = y on imshow;
    # cols_d / W_c → y on image2D = x on imshow.
    # Since we display .T (row→x-like direction), overlay uses rows as x, cols as y.
    ax6b.imshow(ori_crop.T, cmap="gray", interpolation="nearest", aspect="auto")
    # Sort diagonal indices by row for a connected line (matching R's lines())
    sort_ord = np.argsort(indices[:, 0])
    diag_rows = indices[sort_ord, 0]
    diag_cols = indices[sort_ord, 1]
    ax6b.plot(diag_rows, diag_cols, color="red", linewidth=2)
    ax6b.set_title("With Diagonal", fontsize=11)
    ax6b.set_xticks([])
    ax6b.set_yticks([])

    fig6.tight_layout()
    p6 = figures_out / "watershed_simulation_diagonal_overlay.png"
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved: {p6.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
