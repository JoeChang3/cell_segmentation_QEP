#### Workflow Visualization ####
#
# R source: Methods_Workflow/Workflow_Visualization.R
#
# Produces a composite figure illustrating the full GP segmentation pipeline
# applied to a single cropped test image:
#
#   Step 0  — Original image (full)
#   Step 1  — Original image split into 4 quadrants (TL, TR, BL, BR)
#   Step 2  — GP posterior predictive mean for each quadrant
#   Step 3  — Binary threshold applied to each quadrant's predmean
#   Step 4  — Binary image stitched back together (full)
#   Step 5  — Euclidean distance map of the stitched binary
#             (R comment: "not displayed in workflow figure but used for watershed")
#   Final   — GP masks produced by watershed, rendered with distinct colors
#
# R note: the R script had no explicit save calls; it displayed plots
# interactively in sequence. Python consolidates all panels into one
# composite PNG saved to results/figures/.
#
# Layout: 4 rows × 4 columns
#   column 0  : full-image stages (Original → Stitched Binary → Distance Map → GP Masks)
#   column 1  : per-quadrant Step 1 (cropped original)
#   column 2  : per-quadrant Step 2 (GP predmean)
#   column 3  : per-quadrant Step 3 (binary threshold)
#   rows 0–3  : TL, TR, BL, BR quadrant order
#
# Path conventions
#   data/cropped_pieces/  — source image
#   results/figures/      — output figure

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt

from py_core.Modified_Functions_RGasp import generate_gp_masks_test


# -----------------------------------------------
# Paths  (all project-root-relative)
# -----------------------------------------------
# experiments/visualizations/ -> experiments/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "cropped_pieces"
RESULTS_DIR  = PROJECT_ROOT / "results"

# R: file_path <- ".../cropped_pieces/cropped_img_1.jpg"
FILE_PATH = DATA_DIR / "cropped_img_1.jpg"


# -----------------------------------------------
# make_distinct_cmap
# -----------------------------------------------
def make_distinct_cmap(n_labels: int) -> mcolors.ListedColormap:
    """
    R: col = c("white", distinctColorPalette(n-1))
    Approximation: R uses the 'randomcoloR' package for maximally distinct
    colors. Python generates evenly spaced hues in HSV space, which gives
    visually distinct colors for typical cell counts (< ~50).
    The first entry is always white (background label 0).
    """
    if n_labels <= 1:
        return mcolors.ListedColormap(["white"])

    n_cells = n_labels - 1   # exclude background
    # Evenly spaced hues, full saturation and value
    hues = np.linspace(0.0, 1.0, n_cells, endpoint=False)
    cell_colors = [mcolors.hsv_to_rgb((h, 0.85, 0.9)) for h in hues]

    return mcolors.ListedColormap(["white"] + cell_colors)


# -----------------------------------------------
# main
# -----------------------------------------------
def main() -> None:
    figures_out = RESULTS_DIR / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)

    # ---- Load original image ----
    # R: img <- image_read(file_path)
    #    ori_img_matrix <- as.numeric(img[[1]])[,,1]
    raw = imageio.imread(str(FILE_PATH))
    ori_img = raw[..., 0].astype(np.float64) if raw.ndim == 3 else raw.astype(np.float64)

    # ---- Run GP pipeline ----
    # R: gp_masks_result <- generate_GP_Masks_test(file_path)
    # R default nugget=F; Python default nugget=True, so pass nugget=False to match R.
    print(f"Running GP segmentation on {FILE_PATH.name} ...")
    result     = generate_gp_masks_test(str(FILE_PATH), nugget=False)
    gp_masks   = result.gp_masks              # R: GP_masks
    predmean   = result.combined_predmean     # R: test = gp_masks_result$combined_predmean
    binary     = result.combined_thresholded1 # R: bin = gp_masks_result$combined_thresholded1

    # Shared colour scale — R: zlim=c(0, max(as.vector(test)))
    vmax = float(predmean.max())

    # ---- Distance map ----
    # R: distmap(bin)  (EBImage::distmap = Euclidean distance transform)
    # Approximation: scipy distance_transform_edt matches EBImage::distmap for binary inputs.
    dist_map = distance_transform_edt(binary > 0)

    # ---- Quadrant split ----
    # R: 1:(nrow/2)  and  (nrow/2):nrow  — 1-indexed, R's ':' includes both endpoints
    # so there is a 1-row/col overlap at the midpoint; visually negligible.
    H, W   = ori_img.shape
    H2, W2 = H // 2, W // 2

    # (row_slice, col_slice) for TL, TR, BL, BR — matches R's quadrant order in the script
    quads = [
        (slice(None, H2), slice(None, W2)),   # TL
        (slice(None, H2), slice(W2, None)),   # TR
        (slice(H2, None), slice(None, W2)),   # BL
        (slice(H2, None), slice(W2, None)),   # BR
    ]
    quad_labels = ["TL", "TR", "BL", "BR"]

    # ---- Distinct colormap for GP masks ----
    # R: col = c("white", distinctColorPalette(length(unique(GP_masks)) - 1))
    n_labels = len(np.unique(gp_masks))
    gp_cmap  = make_distinct_cmap(n_labels)

    # ---- Build composite figure ----
    # 4 rows × 4 cols:
    #   col 0 : full-image pipeline stages
    #   cols 1-3 : per-quadrant Step1 / Step2 / Step3
    fig = plt.figure(figsize=(16, 16))
    gs  = gridspec.GridSpec(4, 4, figure=fig,
                            hspace=0.04, wspace=0.04,
                            left=0.01, right=0.99,
                            top=0.93, bottom=0.01)

    col_titles = [
        "Pipeline (full image)",
        "Step 1: Cropped Tiles",
        "Step 2: GP Pred Mean",
        "Step 3: Thresholded",
    ]
    row_titles_col0 = [
        "Original Image",
        "Step 4: Stitched Binary",
        "Step 5: Distance Map",
        "Final: GP Masks",
    ]

    # Helper — hide axes, apply shared style
    def _show(ax, img, cmap, vmin=0.0, vmax_=None, label=None):
        kw = dict(cmap=cmap, vmin=vmin,
                  vmax=(vmax_ if vmax_ is not None else img.max()),
                  interpolation="nearest", aspect="auto")
        ax.imshow(img.T, **kw)   # .T because R's image2D displays matrix transposed vs Python imshow
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        if label:
            ax.set_title(label, fontsize=8, pad=2)

    # ---- Column 0: full-image stages ----

    # Row 0 — Original image
    # R: image2D(ori_img_matrix, zlim=c(0,max(test)), ...)
    ax = fig.add_subplot(gs[0, 0])
    _show(ax, ori_img, cmap="viridis", vmin=0.0, vmax_=vmax)
    ax.set_ylabel(row_titles_col0[0], fontsize=8, labelpad=3)

    # Row 1 — Stitched binary (Step 4)
    # R: image2D(bin, col=rev(gray(...)), zlim=c(0,max(test)), ...)
    # Approximation: zlim is [0, vmax] but binary has only 0/1 values; using
    # vmin/vmax=0/1 makes the binary clearly visible (foreground black on white).
    ax = fig.add_subplot(gs[1, 0])
    _show(ax, binary.astype(np.float32), cmap="gray_r", vmin=0.0, vmax_=1.0)
    ax.set_ylabel(row_titles_col0[1], fontsize=8, labelpad=3)

    # Row 2 — Distance map (Step 5)
    # R: image2D(distmap(bin), ...)   — "not displayed in workflow figure but used for watershed"
    ax = fig.add_subplot(gs[2, 0])
    _show(ax, dist_map, cmap="viridis")
    ax.set_ylabel(row_titles_col0[2], fontsize=8, labelpad=3)

    # Row 3 — GP masks, distinct colors (Final)
    # R: image2D(GP_masks, col=c("white", distinctColorPalette(n-1)), ...)
    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(gp_masks.T, cmap=gp_cmap, vmin=0, vmax=n_labels - 1,
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_ylabel(row_titles_col0[3], fontsize=8, labelpad=3)

    # ---- Columns 1–3: per-quadrant panels ----
    for row_idx, (rs, cs) in enumerate(quads):
        qlabel = quad_labels[row_idx]

        # Column 1 — Step 1: cropped original
        # R: image2D(ori_img_matrix[rows, cols], zlim=c(0,max(test)), ...)
        ax = fig.add_subplot(gs[row_idx, 1])
        _show(ax, ori_img[rs, cs], cmap="viridis", vmin=0.0, vmax_=vmax, label=qlabel)

        # Column 2 — Step 2: GP predmean tile
        # R: image2D(test[rows, cols], zlim=c(0,max(test)), ...)
        ax = fig.add_subplot(gs[row_idx, 2])
        _show(ax, predmean[rs, cs], cmap="viridis", vmin=0.0, vmax_=vmax, label=qlabel)

        # Column 3 — Step 3: binary threshold tile
        # R: image2D(bin[rows, cols], col=rev(gray(...)), zlim=c(0,max(test)), ...)
        ax = fig.add_subplot(gs[row_idx, 3])
        _show(ax, binary[rs, cs].astype(np.float32), cmap="gray_r", vmin=0.0, vmax_=1.0,
              label=qlabel)

    # ---- Column headers ----
    for col_idx, title in enumerate(col_titles):
        fig.text(
            (col_index_to_x(col_idx, n_cols=4)),
            0.945,
            title,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    out_path = figures_out / "workflow_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def col_index_to_x(col_idx: int, n_cols: int) -> float:
    """Map a 0-based column index to an approximate figure x-coordinate [0,1]."""
    # GridSpec left=0.01, right=0.99; each column gets equal share
    left, right = 0.01, 0.99
    col_width = (right - left) / n_cols
    return left + col_width * (col_idx + 0.5)


if __name__ == "__main__":
    main()
