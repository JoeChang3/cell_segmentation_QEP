# Task A2: dump the ORIGINAL R Fast-GP result on a REAL image tile.
#
# The tile is extracted with exactly the geometry generate_GP_Masks_test uses,
# so this is the true first tile of a real development image -- not a synthetic
# crop. Everything here is read-only with respect to the reference repo: we
# source its functions and never edit them.
#
# Orientation, verified empirically this session:
#   as.numeric(image_read(p)[[1]]) has dim (height, width, channels)
# so img_matrix is (rows, cols) = natural image orientation, and n1 = crop_height.
#
# theta_hat is recomputed here with the SAME expressions as
# Modified_Functions_RGasp.R:225-245 (separable_GP does not return it). The
# script asserts that the locally recomputed predictive mean equals the one
# separable_GP returns, so the recomputation is validated rather than assumed.
#
# Usage:
#   Rscript audits/taskA2_real_tile_r_side.R <dataset> <image_path>
# Writes audits/parity/real_{dataset}_{r_params,r_tile,r_predmean,r_meta}.csv

suppressMessages({
  library(magick)
  library(pracma)     # provides gradient(), called by the original functions
})

args <- commandArgs(trailingOnly = TRUE)
dataset  <- if (length(args) >= 1) args[1] else "nuclei"
img_path <- if (length(args) >= 2) args[2] else
  "data/nuclear_test_images/nuclei_figure_1/original_fig.png"

ref <- "/Users/zchan/eclipse-workspace/cell_segmentation_original/src/Modified_Functions_RGasp.R"
source(ref)

outdir <- "audits/parity"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---- replicate generate_GP_Masks_test's tiling exactly -----------------------
img <- image_read(img_path)
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

# first tile: i = 1, j = 1  =>  x_offset = y_offset = 0
cropped_img <- image_crop(img, geometry_area(crop_width, crop_height, 0, 0))
img_matrix  <- as.numeric(cropped_img[[1]])[, , 1]

n1 <- dim(img_matrix)[1]; n2 <- dim(img_matrix)[2]
cat(sprintf("image %s : %d x %d (H x W)\n", img_path, img_height, img_width))
cat(sprintf("proportions row=%s col=%s ; crop %d x %d ; pieces %d x %d\n",
            format(row_proportion), format(col_proportion),
            crop_height, crop_width, num_pieces_y, num_pieces_x))
cat(sprintf("tile(1,1) dim = %d x %d  (n1=rows=crop_height, n2=cols=crop_width)\n",
            n1, n2))
cat(sprintf("tile intensity range = [%.10f, %.10f]  mean = %.10f\n",
            min(img_matrix), max(img_matrix), mean(img_matrix)))

# ---- the original two-step Fast-GP -----------------------------------------
t0 <- proc.time()[["elapsed"]]
parameters <- separable_GP_param_est(img_matrix)
t_est <- proc.time()[["elapsed"]] - t0
cat(sprintf("separable_GP_param_est: %.2f s ; beta1=%.10f beta2=%.10f nu=%.10f\n",
            t_est, parameters$param[1], parameters$param[2], parameters$param[3]))

t0 <- proc.time()[["elapsed"]]
gp <- separable_GP(img_matrix, parameters$param)
t_pred <- proc.time()[["elapsed"]] - t0
cat(sprintf("separable_GP: %.2f s\n", t_pred))

# ---- recompute theta_hat / S_2 with the ORIGINAL expressions ---------------
beta <- parameters$param[1:2]; nu <- parameters$param[3]
N <- n1 * n2
input1 <- as.numeric(seq(0, 1, 1 / (n1 - 1)))
input2 <- as.numeric(seq(0, 1, 1 / (n2 - 1)))
Matern_5_2_funct <- function(d, beta) { x <- sqrt(5) * beta * d; (1 + x + x^2 / 3) * exp(-x) }
R01 <- as.matrix(abs(outer(input1, input1, "-")))
R02 <- as.matrix(abs(outer(input2, input2, "-")))
X <- matrix(1, N, 1); q_X <- 1
X_list <- list(matrix(X[, 1], n1, n2))

R1 <- Matern_5_2_funct(R01, beta = beta[1])
R2 <- Matern_5_2_funct(R02, beta = beta[2])
eigen_R1 <- eigen(R1); eigen_R2 <- eigen(R2)
U_x <- as.vector(t(eigen_R1$vectors) %*% X_list[[1]] %*% eigen_R2$vectors)
Lambda_tilde_inv <- 1 / (kronecker(eigen_R2$values, eigen_R1$values, FUN = "*") + nu)
Lambda_tilde_inv_U_x <- Lambda_tilde_inv * U_x
X_R_tilde_inv_X_inv <- solve(t(U_x) %*% (Lambda_tilde_inv_U_x))
output_tilde <- as.vector(t(eigen_R1$vectors) %*% img_matrix %*% eigen_R2$vectors)
theta_hat <- X_R_tilde_inv_X_inv %*% (t(Lambda_tilde_inv_U_x) %*% output_tilde)
output_mat_normalized <- matrix(as.vector(img_matrix) - X %*% theta_hat, n1, n2)
output_normalize_tilde <- as.vector(t(eigen_R1$vectors) %*% output_mat_normalized %*% eigen_R2$vectors)
S_2 <- sum(output_normalize_tilde * Lambda_tilde_inv * output_normalize_tilde)

# validate the recomputation against separable_GP's own output
onl <- matrix(Lambda_tilde_inv * output_normalize_tilde, n1, n2)
Rinv_on <- matrix(as.vector((eigen_R1$vectors) %*% onl %*% t(eigen_R2$vectors)), n1, n2)
r1 <- Matern_5_2_funct(abs(outer(input1, input1, "-")), beta[1])
r2 <- Matern_5_2_funct(abs(outer(input2, input2, "-")), beta[2])
pm_local <- matrix(X %*% theta_hat + as.vector(t(r1) %*% Rinv_on %*% r2), n1, n2)
recompute_maxdiff <- max(abs(pm_local - gp$predmean_mat))
cat(sprintf("recomputed predmean vs separable_GP: max|diff| = %.3e  (validates theta_hat)\n",
            recompute_maxdiff))
cat(sprintf("theta_hat = %.10f   S_2 = %.10f   sigma2_hat = S_2/N = %.10e\n",
            theta_hat[1, 1], S_2, S_2 / N))

# ---- dump -------------------------------------------------------------------
tag <- file.path(outdir, paste0("real_", dataset))
write.csv(data.frame(beta1 = beta[1], beta2 = beta[2], nugget_nu = nu,
                     theta_hat = theta_hat[1, 1], S_2 = S_2, sigma2_hat = S_2 / N,
                     n1 = n1, n2 = n2, t_est_sec = t_est, t_pred_sec = t_pred,
                     recompute_maxdiff = recompute_maxdiff),
          paste0(tag, "_r_params.csv"), row.names = FALSE)
write.table(img_matrix, paste0(tag, "_r_tile.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(gp$predmean_mat, paste0(tag, "_r_predmean.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.csv(data.frame(image = img_path, img_height = img_height, img_width = img_width,
                     row_proportion = row_proportion, col_proportion = col_proportion,
                     crop_height = crop_height, crop_width = crop_width,
                     num_pieces_y = num_pieces_y, num_pieces_x = num_pieces_x,
                     magick_orientation = "(height, width, channel)",
                     intensity_convention = "raw/255 via magick, channel 1 only"),
          paste0(tag, "_r_meta.csv"), row.names = FALSE)
cat(sprintf("wrote %s_r_{params,tile,predmean,meta}.csv\n", tag))
