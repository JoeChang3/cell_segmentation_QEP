# Round 6 SECONDARY: run the paper downstream on a reconstruction supplied from
# Python (used for the robust/multi-start Fast-GP fit, which is NOT the published
# algorithm and is therefore kept out of the primary 2x2 table).
#
# Identical downstream to audits/round6_paper_downstream.R: the supplied
# full-image reconstruction is re-tiled with the same get_proportion geometry,
# then criterion_1 (real RobustGaSP) per tile, outlier-tile handling, stitch,
# EBImage distmap + watershed, eliminate_small_areas(50).
#
# criterion_1 thresholds at `percentage * max(tile)`, so the result is invariant
# to a positive rescaling of intensity; the input is divided by 255 anyway to
# match magick's convention exactly.
#
# Usage: Rscript audits/round6_paper_downstream_supplied.R <dataset> <img> <recon.csv> <outdir> <tag>

.libPaths(c("/Users/zchan/Library/R/arm64/4.5/library", .libPaths()))
suppressMessages({
  library(magick); library(pracma); library(RobustGaSP); library(EBImage)
})

a <- commandArgs(trailingOnly = TRUE)
dataset <- a[1]; img_path <- a[2]; recon_csv <- a[3]; outdir <- a[4]; tag <- a[5]

REPO <- "/Users/zchan/eclipse-workspace/cell_segmentation_original"
source(file.path(REPO, "src", "Modified_Functions_RGasp.R"))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
wr <- function(m, f) write.table(m, file.path(outdir, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)

recon <- as.matrix(read.csv(recon_csv, header = FALSE))
if (max(recon) > 1 + 1e-9) recon <- recon / 255   # match magick raw/255

info <- image_info(image_read(img_path))
img_width <- info$width; img_height <- info$height
stopifnot(nrow(recon) == img_height, ncol(recon) == img_width)

crop_width  <- as.integer(img_width  * get_proportion(img_width))
crop_height <- as.integer(img_height * get_proportion(img_height))
num_pieces_x <- floor(img_width  / crop_width)
num_pieces_y <- floor(img_height / crop_height)
crop_width  <- img_width  %/% num_pieces_x
crop_height <- img_height %/% num_pieces_y

t_all <- proc.time()[["elapsed"]]
processed <- list(); thr_imgs <- list(); pcts <- list(); ncp <- list(); count <- 1
t_thresh <- 0
for (i in 1:num_pieces_x) {
  for (j in 1:num_pieces_y) {
    xo <- (i - 1) * crop_width; yo <- (j - 1) * crop_height
    pm <- recon[yo + 1:crop_height, xo + 1:crop_width]
    processed[[count]] <- pm
    t0 <- proc.time()[["elapsed"]]
    ci <- criterion_1(pm, 0.01, TRUE)
    t_thresh <- t_thresh + (proc.time()[["elapsed"]] - t0)
    thr_imgs[[count]] <- ci$thresholded_image
    ncp[[count]] <- length(unique(as.vector(bwlabel(ci$thresholded_image))))
    pcts[[count]] <- ci$estimated_percentage
    count <- count + 1
  }
}
mcp <- mean(unlist(ncp)); scp <- sd(unlist(ncp))
outliers <- which(abs(unlist(ncp) - mcp) > 2 * scp)
for (oi in outliers) {
  rt <- threshold_image(processed[[oi]], mean(as.numeric(pcts)[-outliers]), FALSE)
  pcts[oi] <- mean(as.numeric(pcts)[-outliers])
  if (sum(as.vector(rt)) > 0.99 * (nrow(rt) * ncol(rt))) {
    rt <- matrix(0, nrow(rt), ncol(rt)); pcts[oi] <- 1
  }
  thr_imgs[[oi]] <- rt
}
cp <- matrix(0, img_height, img_width); cb <- matrix(0, img_height, img_width)
count <- 1
for (i in 1:num_pieces_x) {
  for (j in 1:num_pieces_y) {
    xo <- (i - 1) * crop_width; yo <- (j - 1) * crop_height
    pm <- processed[[count]]; th <- thr_imgs[[count]]
    cp[yo + 1:nrow(pm), xo + 1:ncol(pm)] <- pm
    cb[yo + 1:nrow(th), xo + 1:ncol(th)] <- th
    count <- count + 1
  }
}
t0 <- proc.time()[["elapsed"]]
dm <- distmap(as.Image(cb))
seg <- EBImage::watershed(dm)
raw_lab <- seg@.Data
final <- eliminate_small_areas(raw_lab, 50)
t_ws <- proc.time()[["elapsed"]] - t0
t_total <- proc.time()[["elapsed"]] - t_all

wr(cb, sprintf("%s_combined_binary.csv", tag))
wr(raw_lab, sprintf("%s_watershed_precleanup.csv", tag))
wr(final, sprintf("%s_labels_final.csv", tag))
write.csv(data.frame(dataset = dataset, arm = tag, use_gp = TRUE,
                     recon_source = recon_csv,
                     n_outlier_tiles = length(outliers),
                     fg_fraction = mean(cb),
                     n_labels_precleanup = length(unique(as.vector(raw_lab))) - 1,
                     n_labels_final = length(unique(as.vector(final))) - 1,
                     t_recon_s = NA, t_threshold_s = t_thresh,
                     t_watershed_cleanup_s = t_ws, t_total_s = t_total,
                     cleanup_rule = "eliminate_small_areas(size_threshold=50)"),
          file.path(outdir, sprintf("%s_summary.csv", tag)), row.names = FALSE)
cat(sprintf("  %-22s thresh %5.1fs ws %5.1fs | fg=%.3f labels %d->%d outliers=%d\n",
            tag, t_thresh, t_ws, mean(cb),
            length(unique(as.vector(raw_lab))) - 1,
            length(unique(as.vector(final))) - 1, length(outliers)))
