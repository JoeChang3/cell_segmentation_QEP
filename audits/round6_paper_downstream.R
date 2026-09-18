# Round 6: run the ORIGINAL PAPER DOWNSTREAM in real R, with real RobustGaSP and
# real EBImage. Dumps every intermediate as plain numeric CSV so the current
# Python evaluator can score all arms identically.
#
# Reference repo (READ-ONLY, never modified):
#   /Users/zchan/eclipse-workspace/cell_segmentation_original
#   commit 44714c2e0be958fe796a8fd4bdbc220dae3c23dd
#
# TRACED CALL GRAPH (src/Modified_Functions_RGasp.R::generate_GP_Masks_test, L606)
#   magick::image_read -> get_proportion tiling -> per tile:
#     [GP arms only] separable_GP_param_est on tile (1,1) ONCE, then separable_GP
#     criterion_1(predmean_mat, delta=0.01, nugget=TRUE)
#        candidates = seq(0,1,by=0.01) as PROPORTIONS OF max(tile)
#        diff_pixel_counts = abs(diff(pixel_counts))
#        RobustGaSP: rgasp(percentages[-1], diff_pixel_counts, nugget.est=TRUE)
#                    predict(...)$mean   REPLACES diff_pixel_counts
#        walk forward from which.max until |d[i]-d[i-1]| < 0.05*sd(d)
#        selected percentage = percentages[stable_index + 1]
#        fallback when not found: ALL BACKGROUND, percentage = 1
#     bwlabel -> connected part count per tile
#   outlier tiles: |count - mean| > 2*sd  -> re-threshold at mean of non-outlier
#     percentages; if > 0.99 foreground -> revert to all background, pct = 1
#   stitch into (img_height, img_width)
#   distmap(as.Image(combined_thresholded1))        EBImage, metric="euclidean"
#   EBImage::watershed(dist_map)                     defaults tolerance=1, ext=1
#   eliminate_small_areas(GP_masks_raw, remove_size_threshold=50)
#
# ARMS PRODUCED HERE
#   D_paper_fastgp     literal generate_GP_Masks_test()  -- the published GP arm
#   B_nogp_canonical   CONSTRUCTED ABLATION: identical downstream, but
#                      predmean_mat <- img_matrix (image Fast-GP removed and
#                      NOTHING else changed). This is the clean control for D.
#   B_nogp_literal     literal generate_GP_Masks_test2() from
#                      Nuclear_Real_Analysis/GP_vs_NoGP/ -- the PUBLISHED NoGP
#                      arm. NOTE: it differs from D in THREE ways at once
#                      (no image GP, no rgasp on the criterion curve, and
#                      eliminate_small_areas2 with mean*0.15 / mean*0.05
#                      instead of eliminate_small_areas with 50 / 10).
#                      Reported as a secondary diagnostic, not as the 2x2 B.
#
# Usage: Rscript audits/round6_paper_downstream.R <dataset> <image_path> <outdir>

.libPaths(c("/Users/zchan/Library/R/arm64/4.5/library", .libPaths()))
suppressMessages({
  library(magick); library(pracma); library(RobustGaSP); library(EBImage)
})

args <- commandArgs(trailingOnly = TRUE)
dataset  <- args[1]
img_path <- args[2]
outdir   <- args[3]

REPO <- "/Users/zchan/eclipse-workspace/cell_segmentation_original"
source(file.path(REPO, "src", "Modified_Functions_RGasp.R"))
source(file.path(REPO, "Nuclear_Real_Analysis", "GP_vs_NoGP",
                 "Segmentation_Functions_NoGP (no conflict function names).R"))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("EBImage %s | RobustGaSP %s | magick %s | R %s\n",
            packageVersion("EBImage"), packageVersion("RobustGaSP"),
            packageVersion("magick"), getRversion()))

wr <- function(m, f) write.table(m, file.path(outdir, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)

# ---------------------------------------------------------------------------
# One function that reproduces generate_GP_Masks_test EXACTLY, with a switch for
# whether the image Fast-GP is applied, and full intermediate capture.
# `use_gp=FALSE` changes ONE line (predmean_mat <- img_matrix) and nothing else.
# ---------------------------------------------------------------------------
run_paper_pipeline <- function(file_path, use_gp, tag,
                               delta = 0.01, remove_size_threshold = 50,
                               nugget = TRUE, dump_tile = 1) {
  t_all <- proc.time()[["elapsed"]]
  img <- image_read(file_path)
  info <- image_info(img)
  img_width <- info$width; img_height <- info$height

  row_proportion <- get_proportion(img_height)
  col_proportion <- get_proportion(img_width)
  crop_width  <- as.integer(img_width  * col_proportion)
  crop_height <- as.integer(img_height * row_proportion)
  num_pieces_x <- floor(img_width  / crop_width)
  num_pieces_y <- floor(img_height / crop_height)
  crop_width  <- img_width  %/% num_pieces_x
  crop_height <- img_height %/% num_pieces_y

  processed_images <- list(); thresholded1_images <- list()
  crit_1_opt_thresholds <- list(); connected_parts_count <- list()
  parameters <- NULL; count <- 1
  t_recon <- 0; t_thresh <- 0
  tile_meta <- NULL

  for (i in 1:num_pieces_x) {
    for (j in 1:num_pieces_y) {
      x_offset <- (i - 1) * crop_width
      y_offset <- (j - 1) * crop_height
      cropped_img <- image_crop(img, geometry_area(crop_width, crop_height,
                                                   x_offset, y_offset))
      img_matrix <- as.numeric(cropped_img[[1]])[, , 1]

      if (use_gp) {
        t0 <- proc.time()[["elapsed"]]
        if (i == 1 && j == 1) parameters <- separable_GP_param_est(img_matrix)
        gp <- separable_GP(img_matrix, parameters$param)
        predmean_mat <- gp$predmean_mat
        t_recon <- t_recon + (proc.time()[["elapsed"]] - t0)
      } else {
        predmean_mat <- img_matrix
      }
      processed_images[[count]] <- predmean_mat

      t0 <- proc.time()[["elapsed"]]
      ci <- criterion_1(predmean_mat, delta, nugget)
      t_thresh <- t_thresh + (proc.time()[["elapsed"]] - t0)
      thresholded1_images[[count]] <- ci$thresholded_image
      connected_parts_count[[count]] <-
        length(unique(as.vector(bwlabel(ci$thresholded_image))))
      crit_1_opt_thresholds[[count]] <- ci$estimated_percentage

      # auditable threshold diagnostics for one representative tile
      if (count == dump_tile) {
        pct <- seq(0, 1, by = delta)
        raw_diff <- abs(diff(ci$pixel_counts))
        dm <- rgasp(pct[-1], raw_diff, nugget.est = nugget)
        sm <- predict(dm, testing_input = as.matrix(pct[-1]))
        write.csv(data.frame(percentage = pct[-1],
                             pixel_count = ci$pixel_counts[-1],
                             raw_criterion_curve = raw_diff,
                             rgasp_fitted_curve = sm$mean,
                             rgasp_lower95 = sm$lower95,
                             rgasp_upper95 = sm$upper95),
                  file.path(outdir, sprintf("threshdiag_%s_tile%d.csv", tag, count)),
                  row.names = FALSE)
        write.csv(data.frame(
            selected_percentage = ci$estimated_percentage,
            selected_abs_threshold = ci$estimated_percentage * max(predmean_mat),
            tile_max = max(predmean_mat), tile_min = min(predmean_mat),
            th_stability = 0.05 * sd(sm$mean),
            argmax_index = which.max(sm$mean),
            n_candidates = length(pct),
            rgasp_nugget_est = nugget,
            rgasp_beta_hat = paste(dm@beta_hat, collapse = ";"),
            rgasp_nugget = dm@nugget, rgasp_sigma2 = dm@sigma2_hat),
          file.path(outdir, sprintf("threshdiag_%s_tile%d_selected.csv", tag, count)),
          row.names = FALSE)
        wr(ci$thresholded_image, sprintf("threshdiag_%s_tile%d_binary.csv", tag, count))
        wr(predmean_mat, sprintf("threshdiag_%s_tile%d_predmean.csv", tag, count))
      }
      tile_meta <- rbind(tile_meta, data.frame(
        tile = count, i = i, j = j, x_offset = x_offset, y_offset = y_offset,
        h = nrow(predmean_mat), w = ncol(predmean_mat),
        pct_selected = ci$estimated_percentage,
        abs_threshold = ci$estimated_percentage * max(predmean_mat),
        tile_max = max(predmean_mat),
        n_connected = connected_parts_count[[count]],
        fg_fraction = mean(ci$thresholded_image)))
      count <- count + 1
    }
  }

  # ---- outlier-tile handling (verbatim) ----
  mean_cp <- mean(unlist(connected_parts_count))
  sd_cp <- sd(unlist(connected_parts_count))
  outlier_threshold <- 2
  outliers <- which(abs(unlist(connected_parts_count) - mean_cp) >
                      outlier_threshold * sd_cp)
  reverted <- integer(0)
  for (oi in outliers) {
    rt <- threshold_image(mat = processed_images[[oi]],
                          percentage = mean(as.numeric(crit_1_opt_thresholds)[-outliers]),
                          count = FALSE)
    crit_1_opt_thresholds[oi] <- mean(as.numeric(crit_1_opt_thresholds)[-outliers])
    if (sum(as.vector(rt)) > 0.99 * (nrow(rt) * ncol(rt))) {
      rt <- matrix(0, nrow = nrow(rt), ncol = ncol(rt))
      crit_1_opt_thresholds[oi] <- 1
      reverted <- c(reverted, oi)
    }
    thresholded1_images[[oi]] <- rt
  }

  # ---- stitch ----
  combined_predmean <- matrix(0, nrow = img_height, ncol = img_width)
  combined_thresholded1 <- matrix(0, nrow = img_height, ncol = img_width)
  count <- 1
  for (i in 1:num_pieces_x) {
    for (j in 1:num_pieces_y) {
      x_offset <- (i - 1) * crop_width; y_offset <- (j - 1) * crop_height
      pm <- processed_images[[count]]; th <- thresholded1_images[[count]]
      ph <- nrow(pm); pw <- ncol(pm)
      combined_predmean[y_offset + 1:ph, x_offset + 1:pw] <- pm
      combined_thresholded1[y_offset + 1:ph, x_offset + 1:pw] <- th
      count <- count + 1
    }
  }

  # ---- distance transform + EBImage watershed + cleanup ----
  t0 <- proc.time()[["elapsed"]]
  dist_map <- distmap(as.Image(combined_thresholded1))
  segmented_image <- EBImage::watershed(dist_map)
  GP_masks_raw <- segmented_image@.Data
  GP_masks <- eliminate_small_areas(GP_masks_raw, remove_size_threshold)
  t_ws <- proc.time()[["elapsed"]] - t0
  t_total <- proc.time()[["elapsed"]] - t_all

  wr(combined_predmean,     sprintf("%s_combined_predmean.csv", tag))
  wr(combined_thresholded1, sprintf("%s_combined_binary.csv", tag))
  wr(dist_map@.Data,        sprintf("%s_distmap.csv", tag))
  wr(GP_masks_raw,          sprintf("%s_watershed_precleanup.csv", tag))
  wr(GP_masks,              sprintf("%s_labels_final.csv", tag))
  write.csv(tile_meta, file.path(outdir, sprintf("%s_tile_meta.csv", tag)),
            row.names = FALSE)
  write.csv(data.frame(
      dataset = dataset, arm = tag, use_gp = use_gp,
      img_height = img_height, img_width = img_width,
      crop_height = crop_height, crop_width = crop_width,
      num_pieces_x = num_pieces_x, num_pieces_y = num_pieces_y,
      n_tiles = length(processed_images),
      n_outlier_tiles = length(outliers),
      outlier_tiles = paste(outliers, collapse = ";"),
      n_reverted_to_background = length(reverted),
      cleanup_rule = sprintf("eliminate_small_areas(size_threshold=%d)",
                             remove_size_threshold),
      gp_beta1 = if (use_gp) parameters$param[1] else NA,
      gp_beta2 = if (use_gp) parameters$param[2] else NA,
      gp_nugget = if (use_gp) parameters$param[3] else NA,
      n_labels_precleanup = length(unique(as.vector(GP_masks_raw))) - 1,
      n_labels_final = length(unique(as.vector(GP_masks))) - 1,
      fg_fraction = mean(combined_thresholded1),
      t_recon_s = t_recon, t_threshold_s = t_thresh,
      t_watershed_cleanup_s = t_ws, t_total_s = t_total),
    file.path(outdir, sprintf("%s_summary.csv", tag)), row.names = FALSE)

  cat(sprintf("  %-18s recon %6.1fs thresh %6.1fs ws+clean %5.1fs total %6.1fs | "
              , tag, t_recon, t_thresh, t_ws, t_total))
  cat(sprintf("fg=%.3f labels %d->%d outliers=%d\n",
              mean(combined_thresholded1),
              length(unique(as.vector(GP_masks_raw))) - 1,
              length(unique(as.vector(GP_masks))) - 1, length(outliers)))
  invisible(NULL)
}

# ---------------------------------------------------------------------------
cat(sprintf("\n=== %s : %s ===\n", dataset, img_path))

# D: the published GP arm, verbatim behaviour
run_paper_pipeline(img_path, use_gp = TRUE,  tag = "D_paper_fastgp")

# B: constructed ablation -- image Fast-GP removed, downstream untouched
run_paper_pipeline(img_path, use_gp = FALSE, tag = "B_nogp_canonical")

# B_literal: the PUBLISHED NoGP arm (confounded; secondary diagnostic)
t0 <- proc.time()[["elapsed"]]
lit <- generate_GP_Masks_test2(img_path, nugget = TRUE)
t_lit <- proc.time()[["elapsed"]] - t0
wr(lit$combined_predmean,     "B_nogp_literal_combined_predmean.csv")
wr(lit$combined_thresholded1, "B_nogp_literal_combined_binary.csv")
wr(lit$GP_masks,              "B_nogp_literal_labels_final.csv")
write.csv(data.frame(
    dataset = dataset, arm = "B_nogp_literal", use_gp = FALSE,
    n_labels_final = length(unique(as.vector(lit$GP_masks))) - 1,
    n_outlier_tiles = length(lit$outliers),
    fg_fraction = mean(lit$combined_thresholded1),
    pct_selected = paste(unlist(lit$crit_1_opt_thresholds), collapse = ";"),
    cleanup_rule = "eliminate_small_areas2(mean_obj_size*0.15 / *0.05)",
    criterion_smoother = "NONE (rgasp lines commented out)",
    t_total_s = t_lit),
  file.path(outdir, "B_nogp_literal_summary.csv"), row.names = FALSE)
cat(sprintf("  %-18s total %6.1fs | fg=%.3f labels=%d outliers=%d\n",
            "B_nogp_literal", t_lit, mean(lit$combined_thresholded1),
            length(unique(as.vector(lit$GP_masks))) - 1, length(lit$outliers)))

# ---- ground truth, exactly as the drivers prepare it ----
gt_path <- file.path(dirname(img_path), "original_true_masks.png")
m <- readImage(gt_path)@.Data
if (length(dim(m)) == 3) m <- m[, , 1]
uv <- unique(as.vector(m))
v2i <- setNames(seq(0, length(uv) - 1), uv)
im <- matrix(v2i[as.character(m)], nrow = nrow(m), ncol = ncol(m))
rot <- t(im)[, nrow(im):1]
mir <- rot[, ncol(rot):1]
wr(mir, "GT_paper_orientation.csv")
cat(sprintf("  GT: readImage %s -> driver orientation %s ; equals t(): %s\n",
            paste(dim(im), collapse = "x"), paste(dim(mir), collapse = "x"),
            identical(mir, t(im))))
cat("done\n")
