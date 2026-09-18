# Round 8: the p-cross factorial, run through the ORIGINAL PAPER DOWNSTREAM.
#
# NOTHING IS REFIT. Reconstructions and per-tile selected proportions p come from
# the Round-6 R dumps. criterion_1 / RobustGaSP is NOT re-run: the p values are
# frozen and simply applied to the recipient image.
#
# Four conditions, tile by tile, at identical tile coordinates:
#   RR  raw    image + p_raw      threshold = p_raw    * max(raw    tile)
#   RG  raw    image + p_fastgp   threshold = p_fastgp * max(raw    tile)
#   GR  fastgp image + p_raw      threshold = p_raw    * max(fastgp tile)
#   GG  fastgp image + p_fastgp   threshold = p_fastgp * max(fastgp tile)
# The RECIPIENT image's own max is always used -- we cross the proportion p, not
# an absolute intensity.
#
# Everything after thresholding is the paper downstream, unchanged from Round 6:
#   outlier-tile handling (|n_connected - mean| > 2*sd -> re-threshold at the mean
#   of the non-outlier proportions; >0.99 foreground -> revert to background)
#   -> stitch -> EBImage::distmap -> EBImage::watershed(tolerance=1, ext=1)
#   -> eliminate_small_areas(., 50)
#
# RR must reproduce Round-6 B and GG must reproduce Round-6 D. That is the
# design's self-check and it is asserted downstream in Python.
#
# Usage: Rscript audits/round8_threshold_cross.R <dataset> <img> <r6dir> <outdir>

.libPaths(c("/Users/zchan/Library/R/arm64/4.5/library", .libPaths()))
suppressMessages({library(magick); library(pracma); library(EBImage)})
REPO <- "/Users/zchan/eclipse-workspace/cell_segmentation_original"
source(file.path(REPO, "src", "Modified_Functions_RGasp.R"))

a <- commandArgs(trailingOnly = TRUE)
dataset <- a[1]; img_path <- a[2]; r6dir <- a[3]; outdir <- a[4]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
wr <- function(m, f) write.table(m, file.path(outdir, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)
rd <- function(f) as.matrix(read.csv(file.path(r6dir, f), header = FALSE))

recon <- list(R = rd("B_nogp_canonical_combined_predmean.csv"),
              G = rd("D_paper_fastgp_combined_predmean.csv"))
meta <- list(R = read.csv(file.path(r6dir, "B_nogp_canonical_tile_meta.csv")),
             G = read.csv(file.path(r6dir, "D_paper_fastgp_tile_meta.csv")))
stopifnot(nrow(meta$R) == nrow(meta$G), dim(recon$R) == dim(recon$G))
H <- nrow(recon$R); W <- ncol(recon$R)
cat(sprintf("%s: %dx%d, %d tiles | EBImage %s\n", dataset, H, W, nrow(meta$R),
            packageVersion("EBImage")))

# tile geometry comes straight from Round 6's own record
tiles <- meta$R[, c("tile", "i", "j", "x_offset", "y_offset", "h", "w")]

run_condition <- function(img_key, p_key, tag) {
  M <- recon[[img_key]]
  pv <- meta[[p_key]]$pct_selected
  t_all <- proc.time()[["elapsed"]]

  thr_imgs <- list(); ncp <- list(); pcts <- list(); tmaxes <- numeric(0)
  rows <- NULL
  for (k in seq_len(nrow(tiles))) {
    ro <- tiles$y_offset[k] + 1:tiles$h[k]
    co <- tiles$x_offset[k] + 1:tiles$w[k]
    tile <- M[ro, co]
    tmax <- max(tile, na.rm = TRUE)          # RECIPIENT image's own max
    p <- pv[k]
    thr <- p * tmax                          # paper rule: T = p * max(tile)
    b <- ifelse(tile > thr, 1, 0)            # threshold_image(): strict >
    thr_imgs[[k]] <- b
    ncp[[k]] <- length(unique(as.vector(bwlabel(b))))
    pcts[[k]] <- p
    tmaxes <- c(tmaxes, tmax)
    rows <- rbind(rows, data.frame(
      dataset = dataset, condition = tag, tile = tiles$tile[k],
      p_applied = p, tile_max_recipient = tmax, abs_threshold = thr,
      fg_fraction_tile = mean(b), n_connected_tile = ncp[[k]]))
  }
  # paper outlier-tile handling, verbatim
  mcp <- mean(unlist(ncp)); scp <- sd(unlist(ncp))
  outliers <- which(abs(unlist(ncp) - mcp) > 2 * scp)
  reverted <- integer(0)
  for (oi in outliers) {
    ro <- tiles$y_offset[oi] + 1:tiles$h[oi]
    co <- tiles$x_offset[oi] + 1:tiles$w[oi]
    rt <- threshold_image(M[ro, co], mean(as.numeric(pcts)[-outliers]), FALSE)
    pcts[oi] <- mean(as.numeric(pcts)[-outliers])
    if (sum(as.vector(rt)) > 0.99 * (nrow(rt) * ncol(rt))) {
      rt <- matrix(0, nrow(rt), ncol(rt)); pcts[oi] <- 1
      reverted <- c(reverted, oi)
    }
    thr_imgs[[oi]] <- rt
  }
  cb <- matrix(0, H, W)
  for (k in seq_len(nrow(tiles))) {
    cb[tiles$y_offset[k] + 1:tiles$h[k], tiles$x_offset[k] + 1:tiles$w[k]] <-
      thr_imgs[[k]]
  }
  t0 <- proc.time()[["elapsed"]]
  dm <- distmap(as.Image(cb))
  seg <- EBImage::watershed(dm)                      # tolerance=1, ext=1
  pre <- seg@.Data
  fin <- eliminate_small_areas(pre, 50)
  t_ws <- proc.time()[["elapsed"]] - t0
  t_tot <- proc.time()[["elapsed"]] - t_all

  wr(cb,  sprintf("%s_binary.csv", tag))
  wr(dm@.Data, sprintf("%s_distmap.csv", tag))
  wr(pre, sprintf("%s_precleanup.csv", tag))
  wr(fin, sprintf("%s_labels_final.csv", tag))
  write.csv(rows, file.path(outdir, sprintf("%s_tiles.csv", tag)),
            row.names = FALSE)
  write.csv(data.frame(
      dataset = dataset, condition = tag, image_source = img_key,
      p_source = p_key, n_tiles = nrow(tiles),
      n_outlier_tiles = length(outliers),
      outlier_tiles = paste(outliers, collapse = ";"),
      n_reverted = length(reverted),
      fg_fraction = mean(cb),
      n_fg_components = length(unique(as.vector(bwlabel(cb)))) - 1,
      n_basins_precleanup = length(unique(as.vector(pre))) - 1,
      n_labels_final = length(unique(as.vector(fin))) - 1,
      t_watershed_s = t_ws, t_total_s = t_tot),
    file.path(outdir, sprintf("%s_summary.csv", tag)), row.names = FALSE)
  cat(sprintf("  %-3s img=%s p=%s | fg=%.4f comps=%d basins=%d final=%d outliers=%d (%.1fs)\n",
              tag, img_key, p_key, mean(cb),
              length(unique(as.vector(bwlabel(cb)))) - 1,
              length(unique(as.vector(pre))) - 1,
              length(unique(as.vector(fin))) - 1, length(outliers), t_tot))
}

run_condition("R", "R", "RR")
run_condition("R", "G", "RG")
run_condition("G", "R", "GR")
run_condition("G", "G", "GG")
cat("done\n")
