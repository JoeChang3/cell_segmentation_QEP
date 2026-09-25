# Restore committed R-side parameters; do not rerun the optimizer.
suppressMessages({ library(magick); library(pracma); library(RobustGaSP); library(EBImage) })
a <- commandArgs(trailingOnly=TRUE)
stopifnot(length(a) == 4)
source(file.path(a[1], "src", "Modified_Functions_RGasp.R"))
options(digits=17)
raw <- as.matrix(read.csv(a[2], header=FALSE)) / 255
p <- read.csv(a[3])
params <- c(p$beta1, p$beta2, p$nugget_nu)
h <- nrow(raw); w <- ncol(raw)
nx <- floor(w / as.integer(w * get_proportion(w)))
ny <- floor(h / as.integer(h * get_proportion(h)))
cw <- w %/% nx; ch <- h %/% ny
stopifnot(ch == p$n1, cw == p$n2)
recon <- matrix(0, h, w)
for (i in seq_len(nx)) for (j in seq_len(ny)) {
  rr <- (j-1)*ch + seq_len(ch); cc <- (i-1)*cw + seq_len(cw)
  recon[rr,cc] <- separable_GP(raw[rr,cc], params)$predmean_mat
}
write.table(recon*255, a[4], sep=",", row.names=FALSE, col.names=FALSE)
