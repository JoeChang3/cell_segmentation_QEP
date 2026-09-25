# Adapted from frozen audits/round6_paper_downstream_supplied.R.
# Only paths, explicit units, deterministic RNG initialization, and diagnostics differ.
suppressMessages({
  library(magick); library(pracma); library(RobustGaSP); library(EBImage)
})

a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 6)
dataset <- a[1]; img_path <- a[2]; recon_csv <- a[3]; outdir <- a[4]; tag <- a[5]

REPO <- a[6]
set.seed(1)
options(digits=17)
source(file.path(REPO, "src", "Modified_Functions_RGasp.R"))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
wr <- function(m, f) write.table(m, file.path(outdir, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)

recon <- as.matrix(read.csv(recon_csv, header = FALSE))
stopifnot(all(is.finite(recon)))
recon <- recon / 255  # Explicit input contract: raw intensity units, no clipping

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
initial_pcts <- as.numeric(pcts)
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

tile_meta <- data.frame(tile=seq_along(pcts), p_initial=initial_pcts,
                        p_final=as.numeric(pcts),
                        max_raw_units=vapply(processed, max, numeric(1))*255,
                        components_including_background=as.numeric(ncp),
                        outlier=seq_along(pcts) %in% outliers)
tile_meta$threshold_raw_units <- tile_meta$p_final * tile_meta$max_raw_units
write.csv(tile_meta, file.path(outdir, paste0(tag, "_tiles.csv")), row.names=FALSE)
writeLines(capture.output(sessionInfo()), file.path(outdir, paste0(tag, "_R_session.txt")))
wr(cb, sprintf("%s_combined_binary.csv", tag))
wr(raw_lab, sprintf("%s_watershed_precleanup.csv", tag))
wr(final, sprintf("%s_labels_final.csv", tag))
write.csv(data.frame(dataset = dataset, arm = tag, use_gp = NA,
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
