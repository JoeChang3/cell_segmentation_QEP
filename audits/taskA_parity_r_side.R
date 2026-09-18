# Task A4: run the ORIGINAL real-data Fast-GP functions on a small tile and dump
# fitted parameters + predictive mean, for numerical parity against Python.
# Read-only use of the original repository.
suppressMessages({library(pracma); library(magick)})
ORIG <- "/Users/zchan/eclipse-workspace/cell_segmentation_original"
source(file.path(ORIG, "src/Modified_Functions_RGasp.R"))

args <- commandArgs(trailingOnly=TRUE)
n <- as.integer(args[1]); outdir <- args[2]

# small top-left crop of the real nuclei development image, read exactly as the
# original pipeline reads it: magick -> [0,1] floats, first channel
img <- image_read(file.path(ORIG, "Image_Data/nuclear_test_images/nuclei_figure_1/original_fig.png"))
m <- as.numeric(img[[1]])[,,1]
tile <- m[1:n, 1:n]
cat(sprintf("R: tile %dx%d range [%.6f, %.6f]\n", nrow(tile), ncol(tile), min(tile), max(tile)))

t0 <- Sys.time()
pe <- separable_GP_param_est(tile)
t1 <- Sys.time()
cat(sprintf("R: separable_GP_param_est -> beta1=%.10f beta2=%.10f nu=%.10f  (%.1fs)\n",
            pe$param[1], pe$param[2], pe$param[3], as.numeric(difftime(t1,t0,units="secs"))))

gp <- separable_GP(tile, pe$param)
pm <- gp$predmean_mat
cat(sprintf("R: predmean range [%.6f, %.6f] mean %.6f\n", min(pm), max(pm), mean(pm)))

write.csv(data.frame(beta1=pe$param[1], beta2=pe$param[2], nu=pe$param[3]),
          file.path(outdir, sprintf("r_params_n%d.csv", n)), row.names=FALSE)
write.table(tile, file.path(outdir, sprintf("r_tile_n%d.csv", n)),
            sep=",", row.names=FALSE, col.names=FALSE)
write.table(pm, file.path(outdir, sprintf("r_predmean_n%d.csv", n)),
            sep=",", row.names=FALSE, col.names=FALSE)
cat("R: wrote outputs\n")
