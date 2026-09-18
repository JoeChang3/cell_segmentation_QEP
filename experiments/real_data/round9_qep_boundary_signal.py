"""
Round 9: do q-dependent LOCAL gradient-posterior statistics contain boundary
evidence at adjacent-cell interfaces that the current pipeline MERGES?

TERMINOLOGY (kept conservative throughout, per the preregistration)
  The tested object is "q-dependent local gradient-posterior statistics under the
  current QePyTorch multitask construction". It is NOT called the joint QEP
  posterior boundary probability. The audited multitask sampler draws a per-point
  radius from Chi2(n_tasks), i.e. it factorizes the elliptical structure across
  space, so it is not established to equal the paper's full spatial joint QEP.

MODEL
  Exact regression on VALUE observations only. The gradient posterior is obtained
  analytically from the gradient-augmented kernel's cross-blocks
  (qpytorch.kernels.Matern52KernelGrad; layout verified interleaved
  [f, df/dx1, df/dx2], gradient prior variance 5/(3 l^2)).
  NO zero-valued gradient pseudo-observations are used as data.
  Per query point we build a 3-dimensional MultivariateQExponential over
  (f, df/dx, df/dy) with the exact posterior mean and 3x3 covariance. The event
  dimension d = n_tasks = 3 is FIXED, so the construction is partition-invariant
  by design; Step 3 verifies this empirically anyway.

Hyperparameters are fitted ONCE per dataset at q=2 on a fixed calibration patch
(the 48x48 patch at the image centre), then frozen for q=2 and q=1.5 alike.

Design is preregistered in preregistered_design.json, written before any result.

Usage: python experiments/real_data/round9_qep_boundary_signal.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpytorch
import qpytorch
import qpytorch.kernels as QK
from qpytorch.distributions import MultivariateQExponential

from py_core.segmentation_eval import load_instance_mask

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R5 = os.path.join(_ROOT, "results", "real_cellseg_round5_corrected_baseline_20260917")

Q_PRIMARY = [2.0, 1.5]
Q_SECONDARY = [1.2]
PATCH = 25              # evaluation patch side
QCORE = 9               # central query block side
CALIB = 48              # calibration patch side
MAX_SITES = 60
N_MC = 20000
SEED = 20260918
JITTER = 1e-6


# --------------------------------------------------------------------------
# gradient-augmented posterior
# --------------------------------------------------------------------------
def grad_posterior(train_xy: torch.Tensor, train_y: torch.Tensor,
                   query_xy: torch.Tensor, kern: QK.Matern52KernelGrad,
                   outputscale: float, noise: float, mean_const: float
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact posterior over (f, df/dx, df/dy) at query points, given VALUES only.

    Returns (mean (m,3), cov (m,3,3)) -- the per-point marginal 3x3 blocks.
    """
    n, m = train_xy.shape[0], query_xy.shape[0]
    with torch.no_grad():
        Kfull_tt = kern(train_xy, train_xy).to_dense() * outputscale
        Kfull_qt = kern(query_xy, train_xy).to_dense() * outputscale
        Kfull_qq = kern(query_xy, query_xy).to_dense() * outputscale
    vi_t = torch.arange(0, 3 * n, 3)          # value rows of the train block
    Ktt = Kfull_tt[vi_t][:, vi_t] + noise * torch.eye(n)
    Kqt = Kfull_qt[:, vi_t]                    # (3m, n) all tasks vs train values

    L = torch.linalg.cholesky(Ktt + JITTER * torch.eye(n))
    resid = (train_y - mean_const).unsqueeze(-1)
    alpha = torch.cholesky_solve(resid, L)             # (n,1)
    mu = (Kqt @ alpha).squeeze(-1)                     # (3m,)
    V = torch.cholesky_solve(Kqt.T, L)                 # (n, 3m)

    mean3 = mu.reshape(m, 3).clone()
    mean3[:, 0] += mean_const                          # constant mean on f only
    cov3 = torch.empty(m, 3, 3)
    for i in range(m):
        sl = slice(3 * i, 3 * i + 3)
        prior = Kfull_qq[sl, sl]
        cov3[i] = prior - Kqt[sl] @ V[:, sl]
    # symmetrize + tiny jitter for sampling stability
    cov3 = 0.5 * (cov3 + cov3.transpose(-1, -2))
    cov3 = cov3 + JITTER * torch.eye(3).expand_as(cov3)
    return mean3, cov3


def sample_grad_norm(mean3: torch.Tensor, cov3: torch.Tensor, q: float,
                     n_mc: int, seed: int) -> torch.Tensor:
    """Monte-Carlo ||grad f|| samples from the LOCAL 3-dim q-exponential.

    Default sampling path (rescale=False) is used deliberately: the Task-B audit
    showed rescale=True applies sqrt(a(q, n_points)), the wrong factor for the
    multitask construction, and demonstrably breaks partition invariance there.
    """
    dist = MultivariateQExponential(mean3, cov3, power=torch.tensor(q))
    torch.manual_seed(seed)
    with torch.no_grad(), gpytorch.settings.debug(False):
        s = dist.rsample(torch.Size([n_mc]))           # (n_mc, m, 3)
    return s[..., 1:].pow(2).sum(-1).sqrt()            # (n_mc, m)


# --------------------------------------------------------------------------
def fit_hypers(img: np.ndarray, q: float, tag: str) -> Dict:
    """Fit (lengthscales, outputscale, noise, mean) on the calibration patch.

    Calibration patch = the CALIB x CALIB block at the geometric image centre.
    Uses no ground truth and no region labels.
    """
    H, W = img.shape
    r0, c0 = H // 2 - CALIB // 2, W // 2 - CALIB // 2
    sub = img[r0:r0 + CALIB, c0:c0 + CALIB]
    # subsample to keep the fit cheap and well-conditioned
    step = 2
    sub = sub[::step, ::step]
    h, w = sub.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    X = torch.tensor(np.stack([rr.ravel(), cc.ravel()], 1), dtype=torch.float64)
    Y = torch.tensor(sub.ravel(), dtype=torch.float64)
    Ym, Ys = Y.mean(), Y.std()
    Yn = (Y - Ym) / Ys

    class M(qpytorch.models.ExactQEP):
        def __init__(self, x, y, lik, power):
            super().__init__(x, y, lik)
            self.power = power
            self.mean_module = qpytorch.means.ConstantMean()
            self.covar_module = QK.ScaleKernel(QK.MaternKernel(nu=2.5, ard_num_dims=2))

        def forward(self, x):
            return MultivariateQExponential(self.mean_module(x),
                                            self.covar_module(x), power=self.power)

    P = torch.tensor(q)
    lik = qpytorch.likelihoods.QExponentialLikelihood(power=P)
    mdl = M(X, Yn, lik, P)
    with torch.no_grad():
        mdl.covar_module.base_kernel.lengthscale = torch.tensor([[3.0, 3.0]])
        mdl.covar_module.outputscale = torch.tensor(1.0)
        lik.noise = torch.tensor(0.05)
    mdl.train(); lik.train()
    opt = torch.optim.Adam(mdl.parameters(), lr=0.1)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(lik, mdl)
    for _ in range(150):
        opt.zero_grad()
        loss = -mll(mdl(X), Yn)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ls = mdl.covar_module.base_kernel.lengthscale.squeeze().tolist()
        os_ = float(mdl.covar_module.outputscale)
        nz = float(lik.noise)
    # lengthscale is in units of the subsampled grid -> convert back to pixels
    ls = [float(v) * step for v in ls]
    out = dict(tag=tag, fit_q=q, calib_patch=f"{CALIB}x{CALIB} at image centre",
               calib_subsample_step=step,
               lengthscale_px=ls, outputscale=os_ * float(Ys) ** 2,
               noise=nz * float(Ys) ** 2, mean_const=float(Ym),
               y_scale=float(Ys), n_calib_points=int(X.shape[0]),
               final_neg_mll=float(loss))
    print(f"  hypers[{tag}] lengthscale_px={[round(v,3) for v in ls]} "
          f"outputscale={out['outputscale']:.5f} noise={out['noise']:.5f} "
          f"mean={out['mean_const']:.4f}  (-mll={float(loss):.4f})")
    return out


# --------------------------------------------------------------------------
def build_regions(gt: np.ndarray, pred: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    """Region classes A-D and the adjacent-pair inventory. GT used for LABELLING."""
    st3 = ndi.generate_binary_structure(2, 2)
    ids = np.unique(gt[gt > 0])
    # per-GT assignment to a predicted instance by argmax IoU
    assign = {}
    for g in ids:
        m = gt == g
        lab, cnt = np.unique(pred[m], return_counts=True)
        keep = lab > 0
        assign[int(g)] = int(lab[keep][np.argmax(cnt[keep])]) if keep.any() else 0

    dil = {int(g): ndi.binary_dilation(gt == g, st3) for g in ids}
    boundary_all = np.zeros(gt.shape, bool)
    for g in ids:
        boundary_all |= (dil[int(g)] & ~(gt == int(g)))

    pairs: List[Dict] = []
    iface_A = np.zeros(gt.shape, bool)
    iface_B = np.zeros(gt.shape, bool)
    iface_any = np.zeros(gt.shape, bool)
    iface_owner: Dict[Tuple[int, int], np.ndarray] = {}

    # candidate adjacencies from dilation overlaps
    seen = set()
    for g1 in ids:
        nb = np.unique(gt[dil[int(g1)]])
        for g2 in nb:
            if g2 <= 0 or int(g2) == int(g1):
                continue
            key = (min(int(g1), int(g2)), max(int(g1), int(g2)))
            if key in seen:
                continue
            seen.add(key)
            a, b = key
            band = dil[a] & dil[b] & (gt > 0)
            # drop pixels touching a third instance
            third = np.zeros(gt.shape, bool)
            for g3 in np.unique(gt[band]):
                if int(g3) in (0, a, b):
                    continue
                third |= dil[int(g3)]
            band = band & ~third
            n_if = int(band.sum())
            if n_if < 4:
                continue
            merged = (assign[a] == assign[b]) and assign[a] != 0
            iface_any |= band
            iface_owner[key] = band
            if merged:
                iface_A |= band
            else:
                iface_B |= band
            m1, m2 = gt == a, gt == b
            pairs.append(dict(
                gt_id_1=a, gt_id_2=b, merged=bool(merged),
                pred_id_1=assign[a], pred_id_2=assign[b],
                interface_len_px=n_if, area_1=int(m1.sum()), area_2=int(m2.sum()),
                centroid_r=float(np.argwhere(band)[:, 0].mean()),
                centroid_c=float(np.argwhere(band)[:, 1].mean())))

    gt_fg = gt > 0
    d_bnd = ndi.distance_transform_edt(~boundary_all)
    d_iface = ndi.distance_transform_edt(~iface_any) if iface_any.any() else \
        np.full(gt.shape, np.inf)
    d_cell = ndi.distance_transform_edt(~gt_fg)

    regions = dict(
        A_merged_interface=iface_A,
        B_separated_interface=iface_B,
        C_outer_boundary=(boundary_all & gt_fg & (d_iface >= 4)),
        D_interior=(gt_fg & (d_bnd >= 4)),
        D_background=((~gt_fg) & (d_cell >= 6)),
    )
    return dict(regions=regions, iface_owner=iface_owner, assign=assign), \
        pd.DataFrame(pairs)


def pick_sites(mask: np.ndarray, n_max: int) -> List[Tuple[int, int, int]]:
    """Connected components of `mask`, ranked by size then evenly spaced."""
    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(2, 2))
    if n == 0:
        return []
    sz = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sz)
    if len(order) > n_max:
        order = order[np.linspace(0, len(order) - 1, n_max).astype(int)]
    out = []
    for k in order:
        rc = np.argwhere(lab == k + 1)
        out.append((int(round(rc[:, 0].mean())), int(round(rc[:, 1].mean())),
                    int(sz[k])))
    return out


# --------------------------------------------------------------------------
def partition_invariance(img: np.ndarray, hp: Dict, ds: str) -> List[Dict]:
    """Step 3: re-verify invariance in THIS exact model, before any boundary use."""
    H, W = img.shape
    r0, c0 = H // 2, W // 2
    sub = img[r0:r0 + PATCH, c0:c0 + PATCH]
    h, w = sub.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    Xtr = torch.tensor(np.stack([rr.ravel(), cc.ravel()], 1), dtype=torch.float64)
    Ytr = torch.tensor(sub.ravel(), dtype=torch.float64)
    # 40 fixed query coordinates inside the patch
    qr, qc = np.meshgrid(np.arange(6, 6 + 8), np.arange(6, 6 + 5), indexing="ij")
    Xq = torch.tensor(np.stack([qr.ravel(), qc.ravel()], 1), dtype=torch.float64)
    kern = QK.Matern52KernelGrad(ard_num_dims=2)
    with torch.no_grad():
        kern.lengthscale = torch.tensor([hp["lengthscale_px"]])

    rows = []
    for q in Q_PRIMARY:
        ref = None
        for parts in (1, 2, 5, len(Xq)):
            cs = int(math.ceil(len(Xq) / parts))
            s1, s2, s3 = [], [], []
            for st in range(0, len(Xq), cs):
                xq = Xq[st:st + cs]
                m3, c3 = grad_posterior(Xtr, Ytr, xq, kern, hp["outputscale"],
                                        hp["noise"], hp["mean_const"])
                s1.append(m3[:, 1:].pow(2).sum(-1).sqrt())
                gn = sample_grad_norm(m3, c3, q, 4000, SEED)
                s2.append(gn.mean(0))
                s3.append((gn > 1.0).double().mean(0))
            cat = dict(S1=torch.cat(s1).numpy(), S2=torch.cat(s2).numpy(),
                       S3=torch.cat(s3).numpy())
            if parts == 1:
                ref = cat
            for k, v in cat.items():
                den = float(np.abs(ref[k]).mean()) or 1.0
                dmax = float(np.abs(v - ref[k]).max())
                # MC standard error for the sampled statistics
                se = (float(np.std(v) / math.sqrt(4000)) if k != "S1" else 0.0)
                verdict = ("INVARIANT" if dmax / den < 1e-6 else
                           "within 3 MC SE" if k != "S1" and dmax <= 3 * se + 1e-12
                           else "NOT INVARIANT")
                rows.append(dict(dataset=ds, q=q, statistic=k, n_chunks=parts,
                                 chunk_size=cs, max_abs_diff=dmax,
                                 rel_diff=dmax / den, mc_se=se, verdict=verdict))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--secondary", action="store_true")
    args = ap.parse_args(); out = args.out
    for s in ("figures", "maps", "logs", "tables"):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]

    pair_inv: List[Dict] = []
    reg_rows: List[Dict] = []
    pair_scores: List[Dict] = []
    pinv_rows: List[Dict] = []
    mc_rows: List[Dict] = []
    hyp_rows: List[Dict] = []
    qs = Q_PRIMARY + (Q_SECONDARY if args.secondary else [])

    print("=" * 132)
    print("ROUND 9  q-dependent LOCAL gradient-posterior statistics at merge "
          "interfaces")
    print("  conservative naming: LOCAL construction (per-point 3-dim MVQEP over "
          "(f, df/dx, df/dy)); NOT the joint spatial QEP")
    print("=" * 132)

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        raw = imageio.imread(os.path.join(_ROOT, im["path_image"]))
        raw = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float64)
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        pred = np.load(os.path.join(R5, "instance_masks",
                                    f"{ds}_{name}_raw.npz"))["instance_mask"]
        fastgp = np.load(os.path.join(R5, "masks",
                                      f"{ds}_{name}_paper_fast_gp.npz"))["predmean"]
        fastgp = fastgp.astype(np.float64)
        assert raw.shape == gt.shape == pred.shape == fastgp.shape
        print(f"\n### {ds}/{name}  {raw.shape}  GT={int(len(np.unique(gt[gt>0])))}")

        # ---- Step 2 : frozen hyperparameters (fit once at q=2) -----------
        hp = fit_hypers(raw, 2.0, f"{ds}/frozen_q2")
        hp["dataset"] = ds
        hyp_rows.append(hp)
        kern = QK.Matern52KernelGrad(ard_num_dims=2)
        with torch.no_grad():
            kern.lengthscale = torch.tensor([hp["lengthscale_px"]])

        # ---- Step 3 : partition invariance in this exact model -----------
        pr = partition_invariance(raw, hp, ds)
        pinv_rows += pr
        bad = [r for r in pr if r["verdict"] == "NOT INVARIANT"]
        print(f"  partition invariance: {len(pr)} checks, "
              f"{len(bad)} NOT INVARIANT  "
              f"(max rel diff {max(r['rel_diff'] for r in pr):.2e})")

        # ---- Step 1 : regions + pair inventory ---------------------------
        rg, pdf = build_regions(gt, pred)
        pdf.insert(0, "dataset", ds); pdf.insert(1, "image", name)
        # local contrast + marker count per pair, for the inventory
        mk = np.load(os.path.join(R5, "instance_masks",
                                  f"{ds}_{name}_raw.npz"))
        bin_fg = np.load(os.path.join(R5, "binary_masks",
                                      f"{ds}_{name}_raw.npz"))["binary"] > 0
        extra = []
        for _, p in pdf.iterrows():
            band = rg["iface_owner"][(int(p.gt_id_1), int(p.gt_id_2))]
            m1, m2 = gt == int(p.gt_id_1), gt == int(p.gt_id_2)
            extra.append(dict(
                iface_mean_intensity=float(raw[band].mean()),
                cell_mean_intensity=float(raw[m1 | m2].mean()),
                local_contrast=float(raw[m1 | m2].mean() - raw[band].mean()),
                iface_fg_coverage=float(bin_fg[band].mean()),
                pair_fg_coverage=float(bin_fg[m1 | m2].mean())))
        pdf = pd.concat([pdf.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        pair_inv.append(pdf)
        nm = int(pdf.merged.sum()); ns = int((~pdf.merged).sum())
        print(f"  adjacent GT pairs: {len(pdf)}  merged={nm}  separated={ns}")

        # ---- site selection ---------------------------------------------
        sites: Dict[str, List] = {}
        for cls, m in rg["regions"].items():
            sites[cls] = pick_sites(m, MAX_SITES)
            print(f"    sites[{cls:<22}] {len(sites[cls]):>3}  "
                  f"(pixels {int(m.sum())})")

        # ---- Steps 4 + 7 : statistics per site ---------------------------
        H, W = raw.shape
        gy_raw, gx_raw = np.gradient(raw)
        grad_raw = np.sqrt(gx_raw ** 2 + gy_raw ** 2)
        gy_g, gx_g = np.gradient(fastgp)
        grad_fastgp = np.sqrt(gx_g ** 2 + gy_g ** 2)
        dist_fg = ndi.distance_transform_edt(bin_fg)
        gy_d, gx_d = np.gradient(dist_fg)
        grad_dist = np.sqrt(gx_d ** 2 + gy_d ** 2)

        # pass 1: collect raw MC samples so c can be set from the q=2 pool
        store: Dict[str, Dict] = {}
        pooled_q2: List[np.ndarray] = []
        for cls, sl in sites.items():
            recs = []
            for (r, c, sz) in sl:
                r0 = int(np.clip(r - PATCH // 2, 0, H - PATCH))
                c0 = int(np.clip(c - PATCH // 2, 0, W - PATCH))
                sub = raw[r0:r0 + PATCH, c0:c0 + PATCH]
                rr, cc = np.meshgrid(np.arange(PATCH), np.arange(PATCH),
                                     indexing="ij")
                Xtr = torch.tensor(np.stack([rr.ravel(), cc.ravel()], 1),
                                   dtype=torch.float64)
                Ytr = torch.tensor(sub.ravel(), dtype=torch.float64)
                o = (PATCH - QCORE) // 2
                qr, qc = np.meshgrid(np.arange(o, o + QCORE),
                                     np.arange(o, o + QCORE), indexing="ij")
                Xq = torch.tensor(np.stack([qr.ravel(), qc.ravel()], 1),
                                  dtype=torch.float64)
                gr, gc = (r0 + qr.ravel()), (c0 + qc.ravel())
                m3, c3 = grad_posterior(Xtr, Ytr, Xq, kern, hp["outputscale"],
                                        hp["noise"], hp["mean_const"])
                s1 = m3[:, 1:].pow(2).sum(-1).sqrt().numpy()
                gn = {q: sample_grad_norm(m3, c3, q, N_MC, SEED) for q in qs}
                if 2.0 in gn:
                    pooled_q2.append(gn[2.0].reshape(-1).numpy())
                recs.append(dict(site=(r, c, sz), gr=gr, gc=gc, s1=s1,
                                 gn={q: v for q, v in gn.items()},
                                 base_raw=grad_raw[gr, gc],
                                 base_fastgp=grad_fastgp[gr, gc],
                                 base_dist=grad_dist[gr, gc]))
            store[cls] = dict(recs=recs)

        pool = np.concatenate(pooled_q2)
        cvals = {p: float(np.percentile(pool, p)) for p in (90, 95, 99)}
        print(f"  c thresholds from the q=2 pool: "
              + "  ".join(f"p{p}={v:.4f}" for p, v in cvals.items()))

        # pass 2: reduce to per-site and per-pixel records
        for cls, d in store.items():
            for rec in d["recs"]:
                row = dict(dataset=ds, image=name, region_class=cls,
                           site_r=rec["site"][0], site_c=rec["site"][1],
                           site_size=rec["site"][2], n_query=len(rec["s1"]),
                           S1_mean=float(rec["s1"].mean()),
                           base_grad_raw=float(rec["base_raw"].mean()),
                           base_grad_fastgp=float(rec["base_fastgp"].mean()),
                           base_grad_dist=float(rec["base_dist"].mean()))
                for q in qs:
                    gn = rec["gn"][q]
                    s2 = gn.mean(0).numpy()
                    row[f"S2_mean_q{q}"] = float(s2.mean())
                    row[f"S2_se_q{q}"] = float(
                        (gn.std(0) / math.sqrt(N_MC)).mean())
                    for p, cv in cvals.items():
                        s3 = (gn > cv).double().mean(0).numpy()
                        row[f"S3_p{p}_q{q}"] = float(s3.mean())
                        row[f"S3_p{p}_se_q{q}"] = float(
                            np.sqrt(s3 * (1 - s3) / N_MC).mean())
                reg_rows.append(row)
                mc_rows.append(dict(dataset=ds, region_class=cls,
                                    site_r=rec["site"][0], site_c=rec["site"][1],
                                    n_mc=N_MC, seed=SEED,
                                    **{f"S2_se_q{q}": row[f"S2_se_q{q}"]
                                       for q in qs}))

        # pair-level: map merged/separated interface sites back to pairs
        for _, p in pdf.iterrows():
            band = rg["iface_owner"][(int(p.gt_id_1), int(p.gt_id_2))]
            cls = "A_merged_interface" if p.merged else "B_separated_interface"
            hits = [rec for rec in store[cls]["recs"]
                    if band[int(np.clip(rec["site"][0], 0, H - 1)),
                            int(np.clip(rec["site"][1], 0, W - 1))]]
            if not hits:
                continue
            rec = hits[0]
            sel = band[rec["gr"], rec["gc"]]
            if sel.sum() < 3:
                continue
            row = dict(dataset=ds, image=name, gt_id_1=int(p.gt_id_1),
                       gt_id_2=int(p.gt_id_2), merged=bool(p.merged),
                       interface_len_px=int(p.interface_len_px),
                       local_contrast=float(p.local_contrast),
                       n_query_on_band=int(sel.sum()),
                       S1=float(rec["s1"][sel].mean()),
                       base_grad_raw=float(rec["base_raw"][sel].mean()),
                       base_grad_fastgp=float(rec["base_fastgp"][sel].mean()),
                       base_grad_dist=float(rec["base_dist"][sel].mean()))
            for q in qs:
                gn = rec["gn"][q]
                row[f"S2_q{q}"] = float(gn.mean(0).numpy()[sel].mean())
                for pp, cv in cvals.items():
                    row[f"S3_p{pp}_q{q}"] = float(
                        (gn > cv).double().mean(0).numpy()[sel].mean())
            pair_scores.append(row)

        # rank / rescaling test over ALL query points of ALL sites
        allq = {q: [] for q in qs}
        for cls, d in store.items():
            for rec in d["recs"]:
                for q in qs:
                    allq[q].append(rec["gn"][q].mean(0).numpy())
        vecs = {q: np.concatenate(v) for q, v in allq.items()}
        rho = float(spearmanr(vecs[1.5], vecs[2.0]).statistic)
        ratio = float(np.median(vecs[1.5] / np.maximum(vecs[2.0], 1e-300)))
        a_pred = math.sqrt(
            math.exp((2 / 1.5) * math.log(2) + math.lgamma(1.5 + 2 / 1.5)
                     - math.log(3) - math.lgamma(1.5))
            / 1.0)
        print(f"  GLOBAL-RESCALING TEST over {len(vecs[2.0])} query points:")
        print(f"    Spearman rho(S2 q=1.5, S2 q=2) = {rho:.8f}")
        print(f"    median ratio S2(1.5)/S2(2)     = {ratio:.6f}   "
              f"sqrt(a(1.5,3)/a(2,3)) predicted = {a_pred:.6f}")
        print(f"    IQR of the ratio               = "
              f"{np.percentile(vecs[1.5]/np.maximum(vecs[2.0],1e-300),[25,75])}")
        json.dump(dict(dataset=ds, spearman_rho_S2=rho, median_ratio=ratio,
                       predicted_sqrt_a_ratio=a_pred,
                       n_query_points=int(len(vecs[2.0])),
                       c_thresholds=cvals),
                  open(os.path.join(out, "tables",
                                    f"{ds}_rescaling_test.json"), "w"), indent=2)
        np.savez_compressed(os.path.join(out, "maps", f"{ds}_query_scores.npz"),
                            **{f"S2_q{q}": vecs[q] for q in qs})

    pd.concat(pair_inv, ignore_index=True).to_csv(
        os.path.join(out, "adjacent_pair_inventory.csv"), index=False)
    rdf = pd.DataFrame(reg_rows)
    rdf.to_csv(os.path.join(out, "boundary_region_metrics.csv"), index=False)
    pdf2 = pd.DataFrame(pair_scores)
    pdf2.to_csv(os.path.join(out, "pair_level_scores.csv"), index=False)
    pd.DataFrame(pinv_rows).to_csv(
        os.path.join(out, "partition_invariance.csv"), index=False)
    pd.DataFrame(mc_rows).to_csv(os.path.join(out, "mc_diagnostics.csv"),
                                 index=False)
    json.dump(dict(
        hyperparameters=hyp_rows, q_values=qs, patch=PATCH, query_core=QCORE,
        n_mc=N_MC, seed=SEED, max_sites_per_class=MAX_SITES,
        model="exact regression on VALUES only; gradient posterior analytic from "
              "Matern52KernelGrad cross-blocks; per-point 3-dim MVQEP over "
              "(f, df/dx, df/dy); no zero-gradient pseudo-observations",
        construction_caveat="LOCAL construction. The audited multitask sampler "
                            "draws a per-point radius from Chi2(n_tasks), so this "
                            "is not established to equal the paper's full spatial "
                            "joint QEP.",
        rescale_kwarg_used=False,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         gpytorch=gpytorch.__version__,
                         qpytorch=qpytorch.__version__, numpy=np.__version__,
                         platform=platform.platform()),
    ), open(os.path.join(out, "model_config.json"), "w"), indent=2)

    # ---- Step 5 : contrasts, pair-level primary -------------------------
    def cohend(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        s = math.sqrt(((len(a) - 1) * a.var(ddof=1) +
                       (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")

    def auc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        if not len(pos) or not len(neg):
            return float("nan")
        r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                     / (len(pos) * len(neg)))

    stat_cols = ["S1"] + [f"S2_q{q}" for q in qs] + \
        [f"S3_p{p}_q{q}" for q in qs for p in (90, 95, 99)] + \
        ["base_grad_raw", "base_grad_fastgp", "base_grad_dist"]
    contrast_rows: List[Dict] = []
    print("\n" + "=" * 132)
    print("STEP 5  DISCRIMINATION  (primary unit = cell PAIR; interior controls "
          "from site-level table)")
    print("=" * 132)
    for ds in sorted(pdf2["dataset"].unique()):
        pm = pdf2[(pdf2.dataset == ds) & pdf2.merged]
        ps = pdf2[(pdf2.dataset == ds) & ~pdf2.merged]
        inter = rdf[(rdf.dataset == ds) & (rdf.region_class == "D_interior")]
        outer = rdf[(rdf.dataset == ds) & (rdf.region_class == "C_outer_boundary")]
        print(f"\n  {ds}   merged pairs={len(pm)}  separated pairs={len(ps)}  "
              f"interior sites={len(inter)}  outer sites={len(outer)}")
        print(f"    {'statistic':<22}{'merged':>10}{'separat':>10}{'interior':>10}"
              f"{'contrast':>10}{'cohen d':>9}{'AUC':>8}")
        for sc in stat_cols:
            icol = {"S1": "S1_mean", "base_grad_raw": "base_grad_raw",
                    "base_grad_fastgp": "base_grad_fastgp",
                    "base_grad_dist": "base_grad_dist"}.get(sc)
            if icol is None:
                icol = sc.replace("S2_q", "S2_mean_q")
            if icol not in inter.columns or sc not in pdf2.columns:
                continue
            mv, sv = pm[sc].values, ps[sc].values
            iv = inter[icol].values
            con = float(np.mean(mv) - np.mean(iv)) if len(mv) and len(iv) else np.nan
            d = cohend(mv, iv); a = auc(mv, iv)
            contrast_rows.append(dict(
                dataset=ds, statistic=sc, mean_merged=float(np.mean(mv)),
                mean_separated=float(np.mean(sv)) if len(sv) else np.nan,
                mean_interior=float(np.mean(iv)),
                mean_outer=float(outer[icol].mean()) if icol in outer else np.nan,
                contrast_merged_minus_interior=con, cohen_d=d, auc=a,
                contrast_merged_minus_separated=(
                    float(np.mean(mv) - np.mean(sv)) if len(sv) else np.nan),
                n_merged=len(mv), n_separated=len(sv), n_interior=len(iv)))
            print(f"    {sc:<22}{np.mean(mv):>10.4f}"
                  f"{(np.mean(sv) if len(sv) else np.nan):>10.4f}"
                  f"{np.mean(iv):>10.4f}{con:>10.4f}{d:>9.3f}{a:>8.4f}")
    pd.DataFrame(contrast_rows).to_csv(
        os.path.join(out, "tables", "discrimination.csv"), index=False)

    print(f"\nWrote outputs into {out}")


if __name__ == "__main__":
    main()
