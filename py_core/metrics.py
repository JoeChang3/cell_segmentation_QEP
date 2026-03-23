
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

import imageio.v2 as imageio
import tifffile as tiff
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries
from py_core.Modified_Functions_RGasp import generate_gp_masks_test


# -----------------------------
# Config
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/Users/zchan/Desktop/projects/Cell_Seg_QEP/Image_Data")

FOLDERS = [
    DATA_ROOT / "whole_cell_test_images" / f"whole_cell_figure_{i}"
    for i in range(1, 6)
]
SKIP = True  # same as R: skip computing, read IoU csvs directly

# IoU thresholds in R: seq(0.5, 0.80, by=0.05)
AP_THRESHOLDS = np.round(np.arange(0.50, 0.80 + 1e-9, 0.05), 2)

def process_image_mask(image_path: str) -> np.ndarray:
    """
    R version:
      imageJ_mask <- readImage(path); imageJ_mask <- imageJ_mask@.Data
      unique values -> map to 0..K-1
      rotated_mask <- t(integer_mask)[, nrow:1]
      mirrored_mask <- rotated_mask[, ncol:1]
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(image_path)
    else:
        arr = imageio.imread(image_path)

    # Ensure 2D
    if arr.ndim == 3:
        # if RGB, take first channel (matches your earlier approach)
        arr = arr[..., 0]

    # Map unique values -> consecutive integers starting from 0
    unique_vals = np.unique(arr)
    val_to_int = {v: i for i, v in enumerate(unique_vals)}
    integer_mask = np.vectorize(val_to_int.get)(arr).astype(np.int32)

    # Rotate: t(mask)[, nrow:1]  -> transpose then reverse columns (because nrow becomes columns after transpose)
    rotated = integer_mask.T[:, ::-1]

    # Mirror: rotated[, ncol:1] -> reverse columns
    mirrored = rotated[:, ::-1]

    return mirrored


# -----------------------------
# IoU + AP
# -----------------------------
def compute_ious(true_mask: np.ndarray, pred_mask: np.ndarray) -> pd.DataFrame:
    """
    Return IoU matrix with rows=true labels, cols=pred labels.
    Only labels > 0.
    """
    true_labels = np.unique(true_mask[true_mask > 0])
    pred_labels = np.unique(pred_mask[pred_mask > 0])

    # Edge cases
    if true_labels.size == 0 or pred_labels.size == 0:
        df = pd.DataFrame(
            np.zeros((len(true_labels), len(pred_labels))),
            index=true_labels.astype(int),
            columns=pred_labels.astype(int),
        )
        return df

    true_regions = [(true_mask == lab) for lab in true_labels]
    pred_regions = [(pred_mask == lab) for lab in pred_labels]

    true_area = np.array([r.sum() for r in true_regions], dtype=np.float64)
    pred_area = np.array([r.sum() for r in pred_regions], dtype=np.float64)

    ious = np.zeros((len(true_labels), len(pred_labels)), dtype=np.float64)

    for i, tr in enumerate(true_regions):
        # compute intersection with every pred region
        for j, pr in enumerate(pred_regions):
            inter = np.logical_and(tr, pr).sum()
            union = true_area[i] + pred_area[j] - inter
            ious[i, j] = (inter / union) if union > 0 else 0.0

    return pd.DataFrame(ious, index=true_labels.astype(int), columns=pred_labels.astype(int))


def compute_ap_from_ious(ious: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Replicate your R logic exactly:
      for each true row, take max IoU; if >= threshold => TP
      FN = unmatched true rows
      FP = unmatched pred cols
      precision = tp / (tp + fp + fn)
    Note: R code marks pred_matched[max_j]=TRUE but does NOT prevent re-using a pred label for multiple trues.
    We keep the same behavior for parity.
    """
    if ious.shape[0] == 0 and ious.shape[1] == 0:
        return {"precision": 0.0, "tp": 0, "fp": 0, "fn": 0}

    tp = 0
    true_matched = np.zeros(ious.shape[0], dtype=bool)
    pred_matched = np.zeros(ious.shape[1], dtype=bool)

    mat = ious.to_numpy()

    for i in range(mat.shape[0]):
        row = mat[i, :]
        if row.size == 0:
            continue
        max_j = int(np.argmax(row))
        row_max = float(row[max_j])
        if row_max >= threshold:
            tp += 1
            true_matched[i] = True
            if pred_matched.size > 0:
                pred_matched[max_j] = True

    fn = int((~true_matched).sum())
    fp = int((~pred_matched).sum())

    precision = tp / (tp + fp + fn) if (tp + fp) > 0 else 0.0
    return {"precision": float(precision), "tp": tp, "fp": fp, "fn": fn}


def compute_ap_table(ious_gp: pd.DataFrame, ious_imagej: pd.DataFrame) -> pd.DataFrame:
    gp_prec = []
    ij_prec = []
    gp_better = []

    for th in AP_THRESHOLDS:
        gp = compute_ap_from_ious(ious_gp, threshold=th)["precision"]
        ij = compute_ap_from_ious(ious_imagej, threshold=th)["precision"]
        gp_prec.append(gp)
        ij_prec.append(ij)
        gp_better.append(gp > ij)

    return pd.DataFrame(
        {
            "Threshold": AP_THRESHOLDS,
            "GP_Method_AP": gp_prec,
            "ImageJ_Method_AP": ij_prec,
            "GP_Better": gp_better,
        }
    )


# -----------------------------
# Boundary figure saving
# -----------------------------
def save_boundary_figures(folder_path: str, ori_img: np.ndarray, gp_masks: np.ndarray,
                          imagej_masks: np.ndarray, true_mask: np.ndarray):
    """
    R uses image2D and points boundary coords.
    Here: show grayscale background, overlay boundary pixels as black points.
    """
    def _save_one(mask: np.ndarray, out_name: str):
        bnd = find_boundaries(mask, mode="outer").astype(np.uint8)
        ys, xs = np.where(bnd == 1)  # row, col

        plt.figure(figsize=(8, 8))
        plt.imshow(ori_img, cmap="gray")
        plt.scatter(xs, ys, s=1, c="black")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(folder_path, out_name), dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    _save_one(gp_masks, "GP_boundaries.png")
    _save_one(imagej_masks, "ImageJ_boundaries.png")
    _save_one(true_mask, "True_boundaries.png")


# -----------------------------
# Save IoUs
# -----------------------------
def save_ious(folder_path: str, true_mask: np.ndarray, gp_masks: np.ndarray, imagej_masks: np.ndarray) -> pd.DataFrame:
    ious_gp = compute_ious(true_mask, gp_masks)
    ious_ij = compute_ious(true_mask, imagej_masks)

    ious_gp.to_csv(os.path.join(folder_path, "ious_gp.csv"), index=True)
    ious_ij.to_csv(os.path.join(folder_path, "ious_imagej.csv"), index=True)

    return compute_ap_table(ious_gp, ious_ij)



# -----------------------------
# Process single image folder
# -----------------------------
def process_image(folder_name: str, base_dir: str) -> pd.DataFrame:
    folder_path = os.path.join(base_dir, folder_name)

    file_path = os.path.join(folder_path, "original_fig.jpg")
    imagej_masks_path = os.path.join(folder_path, "original_ImageJ_masks.tif")
    true_masks_path = os.path.join(folder_path, "original_true_masks.png")

    # Original image to grayscale matrix like: as.numeric(img[[1]])[, , 1]
    ori = imageio.imread(file_path)
    if ori.ndim == 3:
        ori_img = ori[..., 0].astype(np.float64)
    else:
        ori_img = ori.astype(np.float64)

    # Generate GP masks (replace this function)
    gp_masks = generate_gp_masks_test(file_path, nugget=True)

    # Process ImageJ and True Masks
    imagej_masks = process_image_mask(imagej_masks_path)
    true_mask = process_image_mask(true_masks_path)

    # Sanity: shapes should match
    if gp_masks.shape != true_mask.shape:
        raise ValueError(f"GP masks shape {gp_masks.shape} != true mask shape {true_mask.shape}. "
                         "Ensure your GP/QEP pipeline outputs the aligned mask shape.")

    # Save boundary overlays
    save_boundary_figures(folder_path, ori_img, gp_masks, imagej_masks, true_mask)

    # Compute + save IoUs, return AP table
    ap_table = save_ious(folder_path, true_mask, gp_masks, imagej_masks)
    ap_table["Pair"] = folder_name
    return ap_table


# -----------------------------
# Plotting (matplotlib version of your ggplot2)
# -----------------------------
def make_plots(combined_table: pd.DataFrame, out_dir: str):
    # Recode Pair
    pair_map = {
        "whole_cell_figure_1": "Image 1",
        "whole_cell_figure_2": "Image 2",
        "whole_cell_figure_3": "Image 3",
        "whole_cell_figure_4": "Image 4",
        "whole_cell_figure_5": "Image 5",
    }
    df = combined_table.copy()
    df["Pair"] = df["Pair"].map(pair_map).fillna(df["Pair"])

    # Long format like pivot_longer
    plot_data = pd.concat(
        [
            df[["Threshold", "Pair"]].assign(Method="GP", AP=df["GP_Method_AP"].values),
            df[["Threshold", "Pair"]].assign(Method="ImageJ", AP=df["ImageJ_Method_AP"].values),
        ],
        ignore_index=True,
    )

    # ---------- Boxplot: AP by Threshold, grouped by Method ----------
    thresholds = sorted(plot_data["Threshold"].unique())
    methods = ["GP", "ImageJ"]

    # positions for grouped boxplots
    x = np.arange(len(thresholds))
    width = 0.35

    plt.figure(figsize=(10, 8))
    for mi, method in enumerate(methods):
        data = []
        for th in thresholds:
            vals = plot_data[(plot_data["Threshold"] == th) & (plot_data["Method"] == method)]["AP"].values
            data.append(vals)

        pos = x + (mi - 0.5) * width
        METHOD_COLORS = {
            "GP": "#1f77b4",       
            "ImageJ": "#d62728",   
        }

        bp = plt.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=METHOD_COLORS[method], edgecolor="black"),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        # jitter points
        for i_th, th in enumerate(thresholds):
            vals = plot_data[(plot_data["Threshold"] == th) & (plot_data["Method"] == method)]["AP"].values
            jitter_x = np.random.normal(loc=pos[i_th], scale=0.03, size=len(vals))
            plt.scatter(
            jitter_x,
            vals,
            s=40,
            alpha=0.6,
            color=METHOD_COLORS[method],
            edgecolors="black",
            linewidths=0.3
        )


    plt.xticks(x, [str(t) for t in thresholds], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Threshold", fontsize=20, fontweight="bold")
    plt.ylabel("Average Precision (AP)", fontsize=20, fontweight="bold")
    legend_handles = [
        Patch(facecolor=METHOD_COLORS["GP"], edgecolor="black", label="GP"),
        Patch(facecolor=METHOD_COLORS["ImageJ"], edgecolor="black", label="ImageJ"),
    ]
    plt.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_ap_boxplot_comparison.png"), dpi=300)
    plt.close()

    # ---------- Line plot: each image, both methods ----------
    color_map = {"Image 1": "blue", "Image 2": "green", "Image 3": "purple", "Image 4": "orange", "Image 5": "brown"}
    linestyle_map = {"GP": "-", "ImageJ": "--"}

    plt.figure(figsize=(10, 8))
    for pair in sorted(plot_data["Pair"].unique()):
        for method in methods:
            sub = plot_data[(plot_data["Pair"] == pair) & (plot_data["Method"] == method)].sort_values("Threshold")
            plt.plot(
                sub["Threshold"].values,
                sub["AP"].values,
                linestyle=linestyle_map[method],
                linewidth=2,
                color=color_map.get(pair, "black"),
                label=None,
            )
            plt.scatter(
                sub["Threshold"].values,
                sub["AP"].values,
                s=40,
                color=color_map.get(pair, "black"),
            )

    # Build legends similar to ggplot’s color + linetype legends
    # Legend 1: Images (colors)
    handles_img = []
    labels_img = []
    for pair, c in color_map.items():
        h, = plt.plot([], [], color=c, linewidth=3)
        handles_img.append(h)
        labels_img.append(pair)

    # Legend 2: Methods (linetypes)
    handles_m = []
    labels_m = []
    for method, ls in linestyle_map.items():
        h, = plt.plot([], [], color="black", linestyle=ls, linewidth=3)
        handles_m.append(h)
        labels_m.append(method)

    leg1 = plt.legend(handles_img, labels_img, title="Images", loc="center left",
                      bbox_to_anchor=(1.02, 0.6), frameon=False, fontsize=14, title_fontsize=16)
    plt.gca().add_artist(leg1)
    plt.legend(handles_m, labels_m, title="Method", loc="center left",
               bbox_to_anchor=(1.02, 0.2), frameon=False, fontsize=14, title_fontsize=16)

    plt.xlabel("Threshold", fontsize=18, fontweight="bold")
    plt.ylabel("Average Precision (AP)", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ap_comparison_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    if not SKIP:
        # Run full pipeline and compute IoUs
        all_tables = []
        for folder in FOLDERS:
            ap = process_image(folder, BASE_DIR)
            all_tables.append(ap)
        combined = pd.concat(all_tables, ignore_index=True)
    else:
        # Skip heavy pipeline: read saved IoU matrices and compute AP
        all_tables = []
        for folder in FOLDERS:
            folder_path = os.path.join(BASE_DIR, folder)
            ious_gp = pd.read_csv(os.path.join(folder_path, "ious_gp.csv"), index_col=0)
            ious_ij = pd.read_csv(os.path.join(folder_path, "ious_imagej.csv"), index_col=0)

            ap = compute_ap_table(ious_gp, ious_ij)
            ap["Pair"] = os.path.basename(folder)
            all_tables.append(ap)

        combined = pd.concat(all_tables, ignore_index=True)

    # Save combined table
    combined.to_csv(os.path.join(BASE_DIR, "combined_ap_table.csv"), index=False)

    # Plots
    make_plots(combined, BASE_DIR)

    print("Done.")
    print("Saved: combined_ap_table.csv, mean_ap_boxplot_comparison.png, ap_comparison_plot.png")


if __name__ == "__main__":
    main()
