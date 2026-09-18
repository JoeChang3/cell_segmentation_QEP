# Cell-blob benchmark: does q<2 preserve sharp boundaries better than q=2?

`q=2.0` is the exact Gaussian case (verified: `power=2.0` reproduces gpytorch's `log_prob` and `expected_log_prob` bit-for-bit). `q=3.0` is a mechanism probe outside the standard Q-EP range q in (0,2]. `identity_noisy` = no smoothing at all.

Lower is better for all error columns; higher is better for boundary F1. `sharpness` is |grad pred| / |grad truth| inside the edge band, where 1.0 = edge steepness matched and <1 = blurred.


## inference = exact, sigma = 0.1 (mean of 3 seed(s))

| arm | RMSE | rel L1 | rel L2 | rel Linf | edge error | boundary F1 | interior RMSE | edge/interior | sharpness | lengthscale | learned noise | outputscale/noise | tail improve | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q=3.0 (probe) | 0.0910 | 0.3259 | 0.2714 | 0.4778 | 0.1566 | 1.0000 | 0.0370 | 4.2329 | 1.0524 | 0.0452 | 0.2865 | 3.2766 | -0.0075 | 155.4404 |
| q=2.0 | 0.0911 | 0.3264 | 0.2716 | 0.4764 | 0.1566 | 1.0000 | 0.0372 | 4.2147 | 1.0537 | 0.0459 | 0.0179 | 3.4114 | -0.0021 | 164.6050 |
| q=1.8 | 0.0905 | 0.3265 | 0.2699 | 0.4746 | 0.1552 | 1.0000 | 0.0378 | 4.1094 | 1.0581 | 0.0464 | 0.0070 | 3.8390 | 0.0102 | 184.4100 |
| q=1.5 | 0.0819 | 0.3859 | 0.2441 | 0.3806 | 0.1020 | 1.0000 | 0.0716 | 1.4250 | 1.1596 | 0.0640 | 0.0003 | 727.9177 | 0.0617 | 263.0744 |
| q=1.2 | 0.1022 | 0.3559 | 0.3046 | 0.5206 | 0.1788 | 1.0000 | 0.0357 | 5.0162 | 1.0554 | 0.2692 | 0.0001 | 3125.9306 | 0.0974 | 184.8263 |
| q=1.0 | 0.1040 | 0.3603 | 0.3101 | 0.5133 | 0.1826 | 1.0000 | 0.0351 | 5.2003 | 1.0452 | 0.3117 | 0.0001 | 3490.8097 | -0.0008 | 174.5448 |
| GP control | 0.0910 | 0.3260 | 0.2713 | 0.4767 | 0.1565 | 1.0000 | 0.0371 | 4.2203 | 1.0538 | 0.0457 | 0.0179 | 3.4099 | -0.0069 | 163.5709 |
| identity | 0.1002 | 0.4830 | 0.2988 | 0.3770 | 0.1013 | 0.9912 | 0.0998 | 1.0160 | 1.1859 | n/a | n/a | n/a | n/a | 0.0000 |

**Verdict (sigma=0.1, exact):** best q<2 on edge-band RMSE is q=1.5 at 0.1020, vs q=2.0 at 0.1566 -> **q<2 wins at boundaries**.

## inference = exact, sigma = 0.25 (mean of 3 seed(s))

| arm | RMSE | rel L1 | rel L2 | rel Linf | edge error | boundary F1 | interior RMSE | edge/interior | sharpness | lengthscale | learned noise | outputscale/noise | tail improve | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q=3.0 (probe) | 0.1462 | 0.6991 | 0.4359 | 0.6117 | 0.1696 | 0.8773 | 0.1350 | 1.2561 | 1.2299 | 0.0185 | 0.6487 | 2.2727 | 0.0022 | 158.3066 |
| q=2.0 | 0.1221 | 0.4940 | 0.3641 | 0.5430 | 0.2021 | 1.0000 | 0.0622 | 3.2504 | 0.9779 | 0.0596 | 0.0733 | 1.0819 | -0.0025 | 156.9662 |
| q=1.8 | 0.1232 | 0.4986 | 0.3674 | 0.5453 | 0.2042 | 1.0000 | 0.0624 | 3.2740 | 0.9822 | 0.0700 | 0.0293 | 1.6178 | -0.0044 | 157.1516 |
| q=1.5 | 0.1269 | 0.5741 | 0.3782 | 0.5680 | 0.1776 | 0.9827 | 0.0975 | 1.8230 | 1.1703 | 0.0652 | 0.0034 | 17.1461 | 0.0224 | 184.9377 |
| q=1.2 | 0.1358 | 0.6275 | 0.4050 | 0.5937 | 0.1793 | 0.9666 | 0.1115 | 1.6452 | 1.2338 | 0.1612 | 0.0001 | 2193.7166 | 0.0363 | 227.1078 |
| q=1.0 | 0.1241 | 0.5399 | 0.3701 | 0.5498 | 0.1874 | 0.9972 | 0.0838 | 2.2369 | 1.1123 | 0.2235 | 0.0001 | 1998.2434 | 0.0077 | 197.6147 |
| GP control | 0.1222 | 0.4949 | 0.3642 | 0.5431 | 0.2020 | 1.0000 | 0.0625 | 3.2317 | 0.9790 | 0.0594 | 0.0732 | 1.0913 | -0.0038 | 156.8898 |
| identity | 0.2506 | 1.2074 | 0.7471 | 0.9425 | 0.2533 | 0.4868 | 0.2494 | 1.0160 | 1.5165 | n/a | n/a | n/a | n/a | 0.0000 |

**Verdict (sigma=0.25, exact):** best q<2 on edge-band RMSE is q=1.5 at 0.1776, vs q=2.0 at 0.2021 -> **q<2 wins at boundaries**.

## Overall

q<2 beat q=2 on edge-band RMSE in **2 of 2** (inference, sigma) conditions.

| inference | sigma | best q<2 | edge err (best q<2) | edge err (q=2) | q<2 wins? | sharpness q<2 | sharpness q=2 |
|---|---|---|---|---|---|---|---|
| exact | 0.1 | 1.5 | 0.1020 | 0.1566 | YES | 1.160 | 1.054 |
| exact | 0.25 | 1.5 | 0.1776 | 0.2021 | YES | 1.170 | 0.978 |
