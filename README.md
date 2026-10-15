# Cell segmentation: research checkpoints

The latest research checkpoint on `chatgpt-review-cellseg-20260917` is
[Fast-QEP through 2026-10-15](checkpoints/fast_qep_20261015/README.md).
It includes executable sources, frozen protocols, the supplied original-R
end-to-end review, GP-refit/score diagnostics, and independent-seed validation
of fixed control-variate weights. All numerical gates passed in the latest
validation, but no reliable local QEP learning direction was resolved; this
local training branch is paused. No QEP segmentation advantage is claimed.

Rounds 1–10 remain at `fa746853ba007552bb1d5a43340d91f6796096b1`.
The new checkpoint preserves those historical files and records the limits of
the uploaded Mac review (the complete Mac output directory was not supplied).
See its README for integrity checks, source locations and reproduction commands.

The description below is historical. The early isotropic Python GP is not the
paper Fast-GP; consult the audits before interpreting baseline or QEP claims.
The current application focus is real cell segmentation; linear diffusion is
historical exploratory material.

---

- **`nuclei_wholeCell_QEP.py`**  
  Re-implementation of the cell nuclei segmentation experiment from the paper.  
  This version rewrites the original R code for the lattice-based GP model into Python,  
  introduces the **Q-Exponential Process (QEP)** as a generalization of the Gaussian process,  
  and incorporates a Python translation of the **FMOU** (Functional Mean of OU Process) model  
  derived from the `FastGaSP` R package.

- **`linear_diffusion_QEP.py`**  
  Simulates 1D linear diffusion dynamics using a Crank–Nicolson discretization,  
  reproducing the linear diffusion experiment from the same paper.  
  The Gaussian models in the original R version are replaced by QEP models,  
  allowing flexible tail behavior through the _q-power_ parameter.  
  The file also supports FMOU, PCA, and DMD benchmarks for quantitative comparison.

---

## 🧠 Background

The original work **_“Unsupervised cell segmentation by fast Gaussian Processes”_** demonstrated  
that Gaussian Process regression can recover latent cell structures from microscopy data efficiently  
through separable lattice kernels.  

In this repository:
- The **Gaussian Process (GP)** framework is extended to **Q-Exponential Processes (QEP)**,  
  allowing non-Gaussian heavy-tailed behaviors and robustness to noise.  
- The **FMOU model** from the R package `FastGaSP` is reimplemented in Python and integrated.  
- All code is fully compatible with **PyTorch** and **GPyTorch/QePyTorch**, enabling GPU acceleration.

---

