# Segmentation pipeline without image-GP smoothing, but WITH criterion smoothing.
#
# R source: Nuclear_Real_Analysis/GP_vs_NoGP/
#           Segmentation_Functions_NoImageGP (no conflict function names).R
#
# This file differs from segmentation_no_gp.py in exactly one place:
# the criterion_1_2 function in R.
#
#   NoGP R file:       criterion_1_2 smoothing lines are COMMENTED OUT
#                      → diff_pixel_counts used raw (no smoothing)
#                      → translated as criterion_1_no_smooth (segmentation_no_gp.py)
#
#   NoImageGP R file:  criterion_1_2 smoothing lines are ACTIVE
#                      → diff_pixel_counts <- smoothed$mean  (RGaSP)
#                      → translated here as criterion_1 (Modified_Functions_RGasp.py),
#                        which approximates RGaSP with Gaussian smoothing
#
# All other R functions are byte-for-byte identical to the NoGP file and are
# therefore NOT redefined here.  They are imported from existing modules:
#
#   threshold_image2          -> threshold_image        (Modified_Functions_RGasp)
#   get_proportion2           -> get_proportion         (Modified_Functions_RGasp)
#   find_boundaries2          -> find_boundaries_4n     (Modified_Functions_RGasp)
#   eliminate_small_areas2    -> eliminate_small_areas_relative (segmentation_no_gp)
#   _load_image_gray          -> _load_image_gray       (segmentation_no_gp)
#   _run_watershed            -> _run_watershed         (segmentation_no_gp)
#   _detect_outliers          -> _detect_outliers       (segmentation_no_gp)
#
# New pipeline functions in this module:
#   generate_no_image_gp_masks       (generate_GP_Masks2)
#   generate_no_image_gp_masks_test  (generate_GP_Masks_test2)

import math
import numpy as np
from skimage.measure import label as cc_label

from py_core.Modified_Functions_RGasp import (
    threshold_image,
    criterion_1,           # RGaSP-smoothed criterion — this is the sole difference vs NoGP
    get_proportion,
    find_boundaries_4n,    # noqa: F401 — re-exported for callers that import from here
    GenerateMasksResult,
)
from py_core.segmentation_no_gp import (
    eliminate_small_areas_relative,
    _load_image_gray,
    _run_watershed,
    _detect_outliers,
)


# -----------------------------
# generate_no_image_gp_masks
# -----------------------------
def generate_no_image_gp_masks(
    file_path: str,
    delta: float = 0.01,
    nugget: bool = True,
) -> GenerateMasksResult:
    """
    R: generate_GP_Masks2(file_path, delta=0.01, nugget=T)
         from Segmentation_Functions_NoImageGP.R

    Identical pipeline to generate_no_gp_masks() (segmentation_no_gp.py) EXCEPT
    that criterion_1 (with Gaussian smoothing) is used in place of
    criterion_1_no_smooth.

    Approximation: R applies RGaSP (a 1-D GP) to smooth diff_pixel_counts.
    Python uses Gaussian smoothing (criterion_1 from Modified_Functions_RGasp.py)
    as an approximation.  The smoothed curve shape is similar but not identical.

    Tile layout: ceiling(1 / proportion) tiles per axis (float offsets, no
    last-tile remainder correction).
    """
    img_gray = _load_image_gray(file_path)
    img_height, img_width = img_gray.shape

    row_prop = get_proportion(img_height)
    col_prop = get_proportion(img_width)

    crop_width = img_width * col_prop       # R keeps float here
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

            # NoImageGP: predmean_mat = raw image (no GP fitted to the image)
            predmean_mat = tile

            ori_images.append(tile)
            processed_images.append(predmean_mat)

            # Use criterion_1 (Gaussian-smoothed) — the key difference vs NoGP
            c1 = criterion_1(predmean_mat, delta=delta, nugget=nugget)
            thr_img = c1.thresholded_image
            thresholded1_images.append(thr_img)
            crit_1_opt_thresholds.append(float(c1.estimated_percentage))

            # R: length(unique(as.vector(bwlabel(thresholded1_img))))
            cc = cc_label(thr_img > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))

    # ---- 2) outlier detection (R: sd() uses ddof=1)
    outliers = _detect_outliers(connected_parts_count)

    keep = [k for k in range(len(crit_1_opt_thresholds)) if k not in outliers]
    mean_thr = (
        float(np.mean([crit_1_opt_thresholds[k] for k in keep]))
        if keep
        else float(np.mean(crit_1_opt_thresholds))
    )

    # ---- 3) rethreshold outliers (no 99% foreground guard in generate_GP_Masks2)
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
# generate_no_image_gp_masks_test
# -----------------------------
def generate_no_image_gp_masks_test(
    file_path: str,
    delta: float = 0.01,
    nugget: bool = True,
) -> GenerateMasksResult:
    """
    R: generate_GP_Masks_test2(file_path, delta=0.01, nugget=T)
         from Segmentation_Functions_NoImageGP.R

    Identical pipeline to generate_no_gp_masks_test() (segmentation_no_gp.py)
    EXCEPT that criterion_1 (with Gaussian smoothing) is used in place of
    criterion_1_no_smooth.

    Additional differences vs generate_no_image_gp_masks():
      - Tile count uses floor(size / crop) + integer re-normalisation so tiles
        divide the image exactly.  The last tile absorbs any remainder pixels.
      - Outlier rethresholding includes the ≥99% foreground guard: if the
        rethresholded tile has more than 99% foreground pixels it is reverted
        to all-background and its threshold is set to 1.

    Approximation: same RGaSP → Gaussian smoothing substitution as above.
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

    # R: crop_width <- img_width %/% num_pieces_x
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

            # NoImageGP: predmean_mat = raw image (no GP fitted to the image)
            predmean_mat = tile

            ori_images.append(tile)
            processed_images.append(predmean_mat)

            # Use criterion_1 (Gaussian-smoothed) — the key difference vs NoGP
            c1 = criterion_1(predmean_mat, delta=delta, nugget=nugget)
            thr_img = c1.thresholded_image
            thresholded1_images.append(thr_img)
            crit_1_opt_thresholds.append(float(c1.estimated_percentage))

            cc = cc_label(thr_img > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))

    # ---- 2) outlier detection (R: sd() uses ddof=1)
    outliers = _detect_outliers(connected_parts_count)

    keep = [k for k in range(len(crit_1_opt_thresholds)) if k not in outliers]
    mean_thr = (
        float(np.mean([crit_1_opt_thresholds[k] for k in keep]))
        if keep
        else float(np.mean(crit_1_opt_thresholds))
    )

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
