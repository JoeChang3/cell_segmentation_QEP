#### Nuclear Real Analysis — GP vs ImageJ ####
#
# R source: Nuclear_Real_Analysis/Nuclear_Data_Generate_IoU.R
#
# Runs the GP segmentation pipeline on five nuclear test images, compares the
# resulting masks to ImageJ-produced masks using IoU / Average Precision, and
# saves comparison plots.
#
# Relationship to other nuclear scripts in this directory:
#   nuclear_data_generate_iou_gp_vs_nogp.py      — GP vs NoGP (no ImageJ)
#   nuclear_data_generate_iou_gp_vs_noimagegp.py — GP vs NoImageGP (no ImageJ)
#   nuclear_data_generate_iou.py (this file)      — GP vs ImageJ (loaded from TIF)
#
# Path conventions
#   data/nuclear_test_images/nuclei_figure_{1..5}/  — image inputs + per-image IoU CSVs
#   results/tables/                                 — combined AP table
#   results/figures/                                — summary plots

from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries

from py_core.metrics import (
    process_image_mask,     # R: process_image_mask()
    compute_ious,           # R: compute_ious()
    compute_ap_from_ious,   # R: compute_ap_from_ious()
    AP_THRESHOLDS,          # seq(0.5, 0.80, by=0.05)
)
from py_core.Modified_Functions_RGasp import generate_gp_masks_test


# -----------------------------------------------
# Paths  (all project-root-relative)
# -----------------------------------------------
# experiments/real_data/ -> experiments/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "nuclear_test_images"
RESULTS_DIR  = PROJECT_ROOT / "results"

FOLDERS = [DATA_DIR / f"nuclei_figure_{i}" for i in range(1, 6)]

# R: skip = T
# SKIP=False  -> run full pipeline, compute and save IoU CSVs, then plot
# SKIP=True   -> skip heavy computation, read previously saved IoU CSVs, then plot
SKIP = True


# -----------------------------------------------
# compute_ap_table  (GP vs ImageJ variant)
# -----------------------------------------------
def compute_ap_table(ious_gp: pd.DataFrame, ious_imagej: pd.DataFrame) -> pd.DataFrame:
    """
    R: compute_ap_table(ious_gp, ious_imagej)
    Evaluates AP at each threshold in AP_THRESHOLDS for both methods.
    Column names match R output: 'GP_Method_AP' and 'ImageJ_Method_AP'.
    """
    gp_prec     = []
    imagej_prec = []
    gp_better   = []

    for th in AP_THRESHOLDS:
        gp     = compute_ap_from_ious(ious_gp,    threshold=th)["precision"]
        imagej = compute_ap_from_ious(ious_imagej, threshold=th)["precision"]
        gp_prec.append(gp)
        imagej_prec.append(imagej)
        gp_better.append(gp > imagej)

    return pd.DataFrame({
        "Threshold":        AP_THRESHOLDS,
        "GP_Method_AP":     gp_prec,
        "ImageJ_Method_AP": imagej_prec,
        "GP_Better":        gp_better,
    })


# -----------------------------------------------
# save_boundary_figures
# -----------------------------------------------
def save_boundary_figures(
    folder_path: Path,
    ori_img: np.ndarray,
    gp_masks: np.ndarray,
    imagej_masks: np.ndarray,
    true_mask: np.ndarray,
) -> None:
    """
    R: save_boundary_figures(folder_path, ori_img_matrix, GP_masks, ImageJ_Masks, True_Mask)

    Saves three 800×800 PNGs into folder_path:
      GP_boundaries.png, ImageJ_boundaries.png, True_boundaries.png

    Approximation: R uses image2D() + points() with coordinates scaled to [0,1].
    Python uses imshow + scatter with pixel coordinates directly.
    skimage find_boundaries(mode='outer') is used in place of R's custom 4-connectivity
    find_boundaries(); visual result is equivalent for cell boundary overlays.
    """
    def _save_one(mask: np.ndarray, out_name: str) -> None:
        bnd = find_boundaries(mask, mode="outer").astype(np.uint8)
        ys, xs = np.where(bnd == 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ori_img, cmap="gray")
        ax.scatter(xs, ys, s=1, c="black", linewidths=0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(folder_path / out_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    _save_one(gp_masks,     "GP_boundaries.png")
    _save_one(imagej_masks, "ImageJ_boundaries.png")
    _save_one(true_mask,    "True_boundaries.png")


# -----------------------------------------------
# save_ious
# -----------------------------------------------
def save_ious(
    folder_path: Path,
    true_mask: np.ndarray,
    gp_masks: np.ndarray,
    imagej_masks: np.ndarray,
) -> pd.DataFrame:
    """
    R: save_ious(folder_path, True_Mask, GP_masks, ImageJ_Masks)

    Saves ious_gp.csv and ious_imagej.csv into folder_path alongside the images.
    Filenames match R so previously saved CSVs can be reloaded when SKIP=True.
    Returns the AP comparison table.
    """
    ious_gp     = compute_ious(true_mask, gp_masks)
    ious_imagej = compute_ious(true_mask, imagej_masks)

    print("GP IoUs calculated")
    ious_gp.to_csv(folder_path / "ious_gp.csv",      index=True)

    print("ImageJ IoUs calculated")
    ious_imagej.to_csv(folder_path / "ious_imagej.csv", index=True)

    return compute_ap_table(ious_gp, ious_imagej)


# -----------------------------------------------
# align_mask_to_reference
# -----------------------------------------------
def align_mask_to_reference(mask: np.ndarray, ref: np.ndarray, name: str = "mask") -> np.ndarray:
    """
    Align a predicted mask to the reference mask shape.
    Tries identity first; falls back to transpose if shapes are swapped.
    """
    if mask.shape == ref.shape:
        return mask
    if mask.T.shape == ref.shape:
        print(f"  Aligning {name} by transpose: {mask.shape} -> {mask.T.shape}")
        return mask.T
    raise ValueError(f"Cannot align {name} shape {mask.shape} to reference shape {ref.shape}")


# -----------------------------------------------
# process_image
# -----------------------------------------------
def process_image(folder_path: Path) -> pd.DataFrame:
    """
    R: process_image(folder_name, base_dir)

    Processes one nuclear test image folder:
      1. Load original image (PNG).
      2. Generate GP masks (generate_GP_Masks_test in R).
      3. Load ImageJ masks from original_ImageJ_masks.tif.
      4. Load ground-truth mask from original_true_masks.png.
      5. Save boundary overlay figures.
      6. Compute IoUs, save CSVs, return AP table.

    Note: the ImageJ comparison mask is loaded from a pre-existing TIF file,
    not re-computed.
    """
    file_path         = folder_path / "original_fig.png"        # R: file.path(folder_path, "original_fig.png")
    imagej_masks_path = folder_path / "original_ImageJ_masks.tif"
    true_masks_path   = folder_path / "original_true_masks.png"

    # Load original image (first channel, matching R's as.numeric(img[[1]])[,,1])
    ori = imageio.imread(str(file_path))
    ori_img = ori[..., 0].astype(np.float64) if ori.ndim == 3 else ori.astype(np.float64)

    # Generate GP masks — returns GenerateMasksResult; .gp_masks is the label array
    print(f"  Running GP segmentation on {folder_path.name} ...")
    gp_result = generate_gp_masks_test(str(file_path), nugget=True)
    gp_masks  = gp_result.gp_masks

    # Load ImageJ masks from pre-existing TIF
    imagej_masks = process_image_mask(str(imagej_masks_path))

    # Load ground-truth mask
    true_mask = process_image_mask(str(true_masks_path))

    # Debug shapes
    print(f"  ori_img shape:      {ori_img.shape}")
    print(f"  true_mask shape:    {true_mask.shape}")
    print(f"  gp_masks shape:     {gp_masks.shape}")
    print(f"  imagej_masks shape: {imagej_masks.shape}")

    # Align predicted masks to true mask shape
    gp_masks     = align_mask_to_reference(gp_masks,     true_mask, name="gp_masks")
    imagej_masks = align_mask_to_reference(imagej_masks, true_mask, name="imagej_masks")

    # Save boundary overlay figures into the image folder (matching R)
    save_boundary_figures(folder_path, ori_img, gp_masks, imagej_masks, true_mask)

    # Compute IoUs, save CSVs, return AP table
    ap_table = save_ious(folder_path, true_mask, gp_masks, imagej_masks)
    ap_table["Pair"] = folder_path.name   # R: ap_table$Pair <- folder_name
    return ap_table


# -----------------------------------------------
# make_plots
# -----------------------------------------------
def make_plots(combined_table: pd.DataFrame, out_dir: Path) -> None:
    """
    R: boxplot (plot_box) + line plot (plot_line)

    Saves:
      mean_ap_boxplot_comparison (nuclear GP vs ImageJ).png
      ap_comparison_plot (nuclear GP vs ImageJ).png
    into out_dir.  Filenames include a dataset suffix to avoid collision with
    the whole-cell variant, since both R scripts used the same bare filename and
    relied on different working directories.

    Approximation: R uses ggplot2 theme_classic() with full font/margin control.
    Python uses matplotlib with equivalent sizes.  Jitter is added via
    np.random.normal (same intent as ggplot's geom_jitter(width=0.2)).
    """
    # Recode Pair labels — R: recode(Pair, "nuclei_figure_1" = "Image 1", ...)
    pair_map = {f"nuclei_figure_{i}": f"Image {i}" for i in range(1, 6)}
    df = combined_table.copy()
    df["Pair"] = df["Pair"].map(pair_map).fillna(df["Pair"])

    # Pivot to long form — R: pivot_longer(cols=c(GP_Method_AP, ImageJ_Method_AP), ...)
    plot_data = pd.concat([
        df[["Threshold", "Pair"]].assign(Method="GP",     AP=df["GP_Method_AP"].values),
        df[["Threshold", "Pair"]].assign(Method="ImageJ", AP=df["ImageJ_Method_AP"].values),
    ], ignore_index=True)

    thresholds = sorted(plot_data["Threshold"].unique())
    methods    = ["GP", "ImageJ"]
    # R: scale_fill_manual(values = c("GP" = "blue", "ImageJ" = "red"))
    method_colors = {"GP": "blue", "ImageJ": "red"}

    # ---------- Boxplot (plot_box) ----------
    x     = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    for mi, method in enumerate(methods):
        data = []
        for th in thresholds:
            vals = plot_data[
                (plot_data["Threshold"] == th) & (plot_data["Method"] == method)
            ]["AP"].values
            data.append(vals)

        pos = x + (mi - 0.5) * width
        ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=method_colors[method], edgecolor="black"),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        # geom_jitter(width=0.2, alpha=0.5, size=2)
        for i_th, th in enumerate(thresholds):
            vals = plot_data[
                (plot_data["Threshold"] == th) & (plot_data["Method"] == method)
            ]["AP"].values
            jx = np.random.normal(loc=pos[i_th], scale=0.05, size=len(vals))
            ax.scatter(jx, vals, s=40, alpha=0.5, color=method_colors[method],
                       edgecolors="black", linewidths=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds], fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xlabel("Threshold",              fontsize=24, fontweight="bold")
    ax.set_ylabel("Average Precision (AP)", fontsize=24, fontweight="bold")
    legend_handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=m) for m in methods
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=20,
              title_fontsize=22, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_ap_boxplot_comparison (nuclear GP vs ImageJ).png", dpi=300)
    plt.close(fig)

    # ---------- Line plot (plot_line) ----------
    # R: scale_color_manual for images, scale_linetype_manual for methods
    color_map     = {"Image 1": "blue", "Image 2": "green", "Image 3": "purple",
                     "Image 4": "orange", "Image 5": "brown"}
    linestyle_map = {"GP": "-", "ImageJ": "--"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for pair in sorted(plot_data["Pair"].unique()):
        for method in methods:
            sub = plot_data[
                (plot_data["Pair"] == pair) & (plot_data["Method"] == method)
            ].sort_values("Threshold")
            ax.plot(sub["Threshold"].values, sub["AP"].values,
                    linestyle=linestyle_map[method], linewidth=1.5,
                    color=color_map.get(pair, "black"))
            ax.scatter(sub["Threshold"].values, sub["AP"].values,
                       s=30, color=color_map.get(pair, "black"))

    # Image-colour legend
    img_handles = [plt.Line2D([0], [0], color=c, linewidth=3, label=p)
                   for p, c in color_map.items()]
    # Method-linestyle legend (R: override.aes with black lines)
    mth_handles = [plt.Line2D([0], [0], color="black", linestyle=ls, linewidth=1.5, label=m)
                   for m, ls in linestyle_map.items()]
    leg1 = ax.legend(handles=img_handles, title="Images",  fontsize=20, title_fontsize=20,
                     loc="center left", bbox_to_anchor=(1.02, 0.65), frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=mth_handles, title="Method", fontsize=20, title_fontsize=20,
              loc="center left", bbox_to_anchor=(1.02, 0.25), frameon=False)

    ax.set_xlabel("Threshold",              fontsize=20, fontweight="bold")
    ax.set_ylabel("Average Precision (AP)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "ap_comparison_plot (nuclear GP vs ImageJ).png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------
# main
# -----------------------------------------------
def main() -> None:
    # Ensure output directories exist
    tables_out  = RESULTS_DIR / "tables"
    figures_out = RESULTS_DIR / "figures"
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    if not SKIP:
        # ---- Full pipeline ----
        # R: all_ap_tables <- lapply(folders, process_image, base_dir=base_dir)
        all_tables = []
        for folder_path in FOLDERS:
            print(f"Processing {folder_path.name} ...")
            ap = process_image(folder_path)
            all_tables.append(ap)
        combined = pd.concat(all_tables, ignore_index=True)
    else:
        # ---- Skip heavy pipeline: read saved IoU CSVs ----
        # R: if (skip){ iou_data <- lapply(...read.csv...); combined_table <- bind_rows(iou_data) }
        # Note: R's skip=T runs this block; SKIP=True here does the same.
        all_tables = []
        for folder_path in FOLDERS:
            ious_gp_path     = folder_path / "ious_gp.csv"
            ious_imagej_path = folder_path / "ious_imagej.csv"
            if not ious_gp_path.exists() or not ious_imagej_path.exists():
                raise FileNotFoundError(
                    f"IoU CSVs not found in {folder_path}. "
                    "Run with SKIP=False first to generate them."
                )
            ious_gp     = pd.read_csv(ious_gp_path,      index_col=0)
            ious_imagej = pd.read_csv(ious_imagej_path,   index_col=0)
            ap = compute_ap_table(ious_gp, ious_imagej)
            ap["Pair"] = folder_path.name
            all_tables.append(ap)
        combined = pd.concat(all_tables, ignore_index=True)

    # Save combined AP table — Python saves to results/tables/
    out_csv = tables_out / "combined_ap_table_nuclear_gp_vs_imagej.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.relative_to(PROJECT_ROOT)}")

    # Plots — R: ggsave() to current working directory
    make_plots(combined, figures_out)
    print(f"Saved plots to: {figures_out.relative_to(PROJECT_ROOT)}/")
    print("Done.")


if __name__ == "__main__":
    main()
