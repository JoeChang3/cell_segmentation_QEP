"""Round 10, Stage 0 (Part 5): can DEPTH make the predictive mean q-dependent?

Gate before touching cell data. Matched Deep q=2 vs Deep q=1.5 on a kinked
piecewise-smooth 1-D target, everything identical except `power`. Common random
numbers are used for the MC predictive mean so the q comparison is not swamped
by sampling noise. MC standard error is estimated from repeated independent MC
streams applied to the SAME trained model.

Also runs the SHALLOW ExactQEP control at fixed hyperparameters, where the
predictive mean is expected to be q-invariant -- this is what isolates "depth"
as the cause of any deep q effect.

STOP RULE: if |Delta_mu| is not larger than MC error, stop and report NOT SUPPORTED.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np, pandas as pd, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT) if _ROOT not in sys.path else None
torch.set_default_dtype(torch.float64)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gpytorch, qpytorch
from qpytorch.distributions import MultivariateQExponential
from round10_deepqep_core import (env_info, predict_mean, predict_mean_batched,
                                  train_deep)

SEED, ITERS, NIND, HID = 73, 400, 64, 3
N_MC, N_BATCH = 64, 12


def shallow_control(X, Y, Xq, q):
    """Shallow ExactQEP at FIXED hyperparameters -> mean should be q-invariant."""
    P = torch.tensor(float(q))
    lik = qpytorch.likelihoods.QExponentialLikelihood(power=P)
    class M(qpytorch.models.ExactQEP):
        def __init__(s):
            super().__init__(X, Y, lik); s.power = P
            s.mean_module = qpytorch.means.ConstantMean()
            s.covar_module = qpytorch.kernels.ScaleKernel(
                qpytorch.kernels.MaternKernel(nu=2.5))
        def forward(s, x):
            return MultivariateQExponential(s.mean_module(x), s.covar_module(x),
                                            power=s.power)
    m = M()
    with torch.no_grad():
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.08]])
        m.covar_module.outputscale = torch.tensor(1.0)
        lik.noise = torch.tensor(0.01); m.mean_module.constant.fill_(0.0)
    m.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.debug(False):
        return m(Xq).mean.numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = a.out

    torch.manual_seed(0)
    n = 120
    x = torch.linspace(0, 1, n)
    # kinked piecewise-smooth target
    f = torch.where(x < 0.45, 0.8 * torch.sin(4 * math.pi * x),
                    2.2 * (x - 0.45) - 0.35)
    f = f + torch.where(x > 0.75, torch.full_like(x, -0.9), torch.zeros_like(x))
    Y = f + 0.05 * torch.randn(n)
    X = x.unsqueeze(-1)
    Xq = torch.linspace(0, 1, 201).unsqueeze(-1)

    print("=" * 112)
    print("ROUND 10 STAGE 0 (Part 5)  does DEPTH make the predictive mean q-dependent?")
    print("  matched Deep q=2.0 vs q=1.5; identical seed/inducing/steps; common MC streams")
    print("=" * 112)
    ei = env_info()
    for k in ("qpytorch", "qpytorch_file", "gpytorch", "reference_commit",
              "fork_on_sys_path", "deep_qep_module"):
        print(f"  {k:<20} {ei[k]}")

    rows, mus, mses = [], {}, {}
    for q in (2.0, 1.5):
        model, losses = train_deep(X, Y, q, seed=SEED, iters=ITERS,
                                   hidden_dims=HID, num_inducing=NIND)
        mu, se = predict_mean_batched(model, Xq, n_batches=N_BATCH,
                                      n_samples=N_MC, seed0=10_000)
        mus[q], mses[q] = mu, se
        fin = bool(np.isfinite(mu).all() and np.isfinite(losses[-1]))
        rows.append(dict(arm=f"deep_q{q}", power=q, final_loss=losses[-1],
                         loss_finite=bool(np.isfinite(losses[-1])),
                         pred_finite=bool(np.isfinite(mu).all()),
                         mean_absmax=float(np.abs(mu).max()),
                         mc_se_mean=float(se.mean()), mc_se_max=float(se.max()),
                         n_mc=N_MC, n_batches=N_BATCH, iters=ITERS, seed=SEED))
        print(f"\n  deep q={q}: final ELBO loss={losses[-1]:+.5f} finite={fin}  "
              f"|mu|max={np.abs(mu).max():.5f}  MC SE mean={se.mean():.6f} "
              f"max={se.max():.6f}")

    d = mus[1.5] - mus[2.0]
    se_d = np.sqrt(mses[1.5] ** 2 + mses[2.0] ** 2)
    stat = dict(
        max_abs_delta=float(np.abs(d).max()),
        mean_abs_delta=float(np.abs(d).mean()),
        rmse_between_means=float(np.sqrt((d ** 2).mean())),
        corr_between_means=float(np.corrcoef(mus[1.5], mus[2.0])[0, 1]),
        mc_se_of_delta_mean=float(se_d.mean()),
        mc_se_of_delta_max=float(se_d.max()),
        snr_max=float(np.abs(d).max() / se_d.max()),
        snr_mean=float(np.abs(d).mean() / se_d.mean()),
        spatial_nonuniform_sd_over_mean=float(np.abs(d).std() / np.abs(d).mean()))
    print("\n  DEEP q-EFFECT vs MONTE CARLO ERROR")
    for k, v in stat.items():
        print(f"    {k:<34} {v:.6f}")

    sh = {q: shallow_control(X, Y, Xq, q) for q in (2.0, 1.5)}
    sd = sh[1.5] - sh[2.0]
    print(f"\n  SHALLOW ExactQEP control (fixed hyperparameters):")
    print(f"    max|Delta_mu| shallow = {np.abs(sd).max():.3e}   "
          f"(expected ~0: shallow mean is q-invariant)")
    print(f"    deep / shallow ratio  = "
          f"{(np.abs(d).max()/max(np.abs(sd).max(),1e-300)):.3e}")

    gate = (stat["snr_max"] > 3.0 and stat["snr_mean"] > 1.0)
    print(f"\n  STAGE-0 GATE (q effect must exceed MC error): "
          f"{'PASS' if gate else 'FAIL'}")

    pd.DataFrame(rows).to_csv(os.path.join(out, "stage0_arm_summary.csv"),
                              index=False)
    json.dump(dict(stage0=stat, gate_passed=bool(gate),
                   shallow_max_abs_delta=float(np.abs(sd).max()),
                   env=ei, target="kinked piecewise-smooth 1-D",
                   n_train=n, n_query=int(Xq.shape[0]), iters=ITERS,
                   num_inducing=NIND, hidden_dims=HID, seed=SEED,
                   n_mc=N_MC, n_mc_batches=N_BATCH),
              open(os.path.join(out, "stage0_synthetic_gate.json"), "w"),
              indent=2)
    np.savez_compressed(os.path.join(out, "predictions", "stage0_means.npz"),
                        xq=Xq.squeeze(-1).numpy(), mu_q2=mus[2.0],
                        mu_q15=mus[1.5], se_q2=mses[2.0], se_q15=mses[1.5],
                        shallow_q2=sh[2.0], shallow_q15=sh[1.5],
                        x=x.numpy(), y=Y.numpy())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    ax[0].plot(x, Y, "k.", ms=3, label="data")
    ax[0].plot(Xq.squeeze(-1), mus[2.0], label="deep q=2")
    ax[0].plot(Xq.squeeze(-1), mus[1.5], label="deep q=1.5")
    ax[0].legend(fontsize=8); ax[0].set_title("predictive means", fontsize=9)
    ax[1].plot(Xq.squeeze(-1), d, label="Delta_mu = q1.5 - q2")
    ax[1].fill_between(Xq.squeeze(-1), -2 * se_d, 2 * se_d, alpha=.35,
                       label="+/-2 MC SE")
    ax[1].legend(fontsize=8); ax[1].set_title(
        f"deep q effect vs MC error (SNR_max={stat['snr_max']:.1f})", fontsize=9)
    ax[2].plot(Xq.squeeze(-1), sd)
    ax[2].set_title(f"SHALLOW control Delta_mu (max={np.abs(sd).max():.1e})",
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "stage0_synthetic_gate.png"), dpi=130)
    print(f"\nwrote stage0 outputs into {out}")


if __name__ == "__main__":
    main()
