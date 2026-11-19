suppressPackageStartupMessages(library(EBImage))
a <- commandArgs(trailingOnly=TRUE)
source(a[1])
h <- as.integer(a[4])
w <- as.integer(a[5])
cb <- matrix(as.numeric(readBin(a[2],"raw",h*w)), nrow=h,ncol=w)
raw <- EBImage::watershed(distmap(as.Image(cb)))@.Data
labels <- eliminate_small_areas(raw,50)
writeBin(as.integer(labels),a[3],size=4,endian="little")
