# Cell Segmentation with Q-Exponential Processes (QEP)

This repository provides a Python reimplementation and extension of experiments from:

> *Unsupervised cell segmentation by fast Gaussian Processes*

We reproduce the original lattice-based Gaussian process (GP) models and extend them using the **Q-Exponential Process (QEP)** framework introduced in:

> *Bayesian Regularization of Latent Representation*

---

## 📂 Main Scripts

- **`experiments/real_data/nuclei_wholeCell_gp.py`**  
  Reimplementation of the original cell nuclei / whole-cell experiments using the **lattice-based GP (FastGaSP-style)** model.  
  This replaces the previous incorrect sklearn GP implementation with the correct separable lattice algorithm.

- **`experiments/simulated/linear_diffusion_gp.py`**  
  Simulates 1D linear diffusion dynamics and evaluates the lattice GP model against baselines.

- **`experiments/real_data/nuclei_wholeCell_qep.py`**  
  QEP-based extension of the nuclei / whole-cell experiment using **QPyTorch/QEP models**.

- **`experiments/simulated/linear_diffusion_qep.py`**  
  QEP version of the linear diffusion experiment.

---

## 🧠 Model Overview

### 1. Lattice Gaussian Process (Baseline)

- Uses **separable lattice kernels** (Kronecker structure)
- Computational complexity:  
  \[
  \mathcal{O}(k^3 + n^3) \quad \text{instead of} \quad \mathcal{O}((kn)^3)
  \]
- Matches the original R implementation (`FastGaSP`)

---

### 2. Q-Exponential Process (QEP)

- Implemented via **QPyTorch**
- Generalizes Gaussian processes with a **q-exponential likelihood**
- Enables:
  - Heavy-tailed behavior
  - Robustness to noise and outliers

---

### 3. FMOU (Functional Mean of OU Process)

- Python reimplementation of the **FastGaSP FMOU model**
- Used as a low-rank temporal/spatial baseline
- Includes detailed numerical diagnostics to detect instability

---

### 4. Additional Baselines

- PCA
- DMD (Dynamic Mode Decomposition)

---

## 📊 Current Experimental Results (Summary)

### Linear Diffusion (Simulated)

- Lattice GP (Exp / Matérn) achieves the **lowest RMSE**
- FMOU improves over PCA but is less stable
- DMD performs worst due to conditioning issues
- QEP currently underperforms (likely due to tuning / discretization mismatch)

---

### Cell Nuclei / Whole Cell (Real Data)

- Lattice GP consistently outperforms PCA, FMOU, and DMD
- FMOU works but is weaker than GP
- QEP models currently lag behind GP baselines

---

## ⚠️ Known Issues & Debugging

- FMOU previously produced `NaN` due to indexing bug (fixed)
- Numerical diagnostics added to detect:
  - KF divergence
  - Non-finite values
  - instability in EM updates
- QEP performance requires further tuning (kernel / likelihood mismatch)

---

## 🔧 Implementation Notes

- All lattice GP computations follow the **original R code exactly**
- Grid indexing corrected to **1-based (R-style)** for consistency
- Kernel definitions aligned with original implementation
- Avoids dense GP (`O(N^3)`) computations

---

## 🚀 Future Work

- Improve QEP kernel design on lattice grids
- Investigate discretization vs continuous QEP mismatch
- Add GPU acceleration for large-scale experiments
- Integrate segmentation pipeline downstream

---

## 📦 Dependencies

- NumPy / SciPy
- PyTorch
- GPyTorch / QPyTorch
- pydmd (for DMD baseline)

---

## 📌 Notes

- Large data files (>50MB) are included; consider Git LFS if needed.
