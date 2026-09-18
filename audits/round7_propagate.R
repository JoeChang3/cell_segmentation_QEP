# Round 7 secondary: EBImage-NATIVE seeded segmentation with the SAME explicit
# markers used for B*/D*.
#
# EBImage::watershed(x, tolerance=1, ext=1) has no seed argument, so it cannot
# take external markers. EBImage::propagate(x, seeds, mask, lambda) can. It is a
# DIFFERENT algorithm (Voronoi-like propagation under a distance functional, not
# flooding), so this is reported as an EBImage-native cross-check on the seeding
# intervention, NOT as the paper's watershed.
#
# Everything else stays paper-faithful: the EBImage distmap and the RobustGaSP
# binary foreground come from the Round-6 R dumps, and the cleanup is the paper's
# eliminate_small_areas(., 50).
#
# Usage: Rscript audits/round7_propagate.R <dataset> <cell> <r6dir> <markercsv> <outdir>

.libPaths(c("/Users/zchan/Library/R/arm64/4.5/library", .libPaths()))
suppressMessages({library(EBImage); library(pracma)})
source("/Users/zchan/eclipse-workspace/cell_segmentation_original/src/Modified_Functions_RGasp.R")

a <- commandArgs(trailingOnly = TRUE)
dataset <- a[1]; cell <- a[2]; r6dir <- a[3]; mcsv <- a[4]; outdir <- a[5]
tag <- if (cell == "B") "B_nogp_canonical" else "D_paper_fastgp"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

rd <- function(f) as.matrix(read.csv(file.path(r6dir, f), header = FALSE))
dmap   <- rd(sprintf("%s_distmap.csv", tag))
binary <- rd(sprintf("%s_combined_binary.csv", tag))
seeds  <- as.matrix(read.csv(mcsv, header = FALSE))
stopifnot(dim(dmap) == dim(binary), dim(seeds) == dim(binary))

t0 <- proc.time()[["elapsed"]]
prop <- EBImage::propagate(x = as.Image(dmap), seeds = as.Image(seeds),
                           mask = as.Image(binary))
lab_raw <- prop@.Data
lab_raw[is.na(lab_raw)] <- 0
final <- eliminate_small_areas(round(lab_raw), 50)
dt <- proc.time()[["elapsed"]] - t0

write.table(final, file.path(outdir, sprintf("%s_%s_propagate_labels_final.csv",
                                             dataset, cell)),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.csv(data.frame(dataset = dataset, cell = paste0(cell, "*prop"),
                     method = "EBImage::propagate(distmap, seeds, mask, lambda=1e-04)",
                     n_seeds = length(unique(as.vector(seeds))) - 1,
                     n_labels_precleanup = length(unique(as.vector(round(lab_raw)))) - 1,
                     n_labels_final = length(unique(as.vector(final))) - 1,
                     t_s = dt),
          file.path(outdir, sprintf("%s_%s_propagate_summary.csv", dataset, cell)),
          row.names = FALSE)
cat(sprintf("  %-12s %s*prop  seeds=%d  labels %d -> %d  (%.1fs)\n",
            dataset, cell, length(unique(as.vector(seeds))) - 1,
            length(unique(as.vector(round(lab_raw)))) - 1,
            length(unique(as.vector(final))) - 1, dt))
