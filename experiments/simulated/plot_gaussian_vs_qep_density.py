"""
Gaussian vs Q-Exponential Distributions — density comparison figure.

Density formula (qpytorch MultivariateQExponential, N=1 marginal):

    p(x; mu, sigma, q) = (q/2) * (2*pi)^{-1/2} * sigma^{-1}
                         * r^{(q/2 - 1)/2}
                         * exp(-0.5 * r^{q/2})

    where  r = ((x - mu) / sigma)^2

    q = 2  ->  standard Gaussian N(mu, sigma^2)   [exact: r^0 = 1]
    q < 2  ->  heavier tails; integrable cusp at the mean
    q > 2  ->  lighter tails

Source: qpytorch/distributions/multivariate_qexponential.py (docstring, lines 26-31).
Project default: q_power = 1.5  (branin_qep.py, linear_diffusion_qep*.py).

Outputs
-------
results/gaussian_vs_qep_density_slide.png   <- SLIDE-READY (single panel, log scale)
results/gaussian_vs_qep_density.png         <- technical backup (2-panel linear+log)
results/gaussian_vs_qep_density.pdf         <- PDF of backup

Run:
    MPLBACKEND=Agg python experiments/simulated/plot_gaussian_vs_qep_density.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

os.makedirs("results", exist_ok=True)


# ── density ───────────────────────────────────────────────────────────────────

def qexp_density_1d(x, mu=0.0, sigma=1.0, q=2.0):
    """1D Q-Exponential density per qpytorch parameterisation."""
    r = ((x - mu) / sigma) ** 2
    r = np.where(r < 1e-15, 1e-15, r)        # guard against 0^{negative}
    prefactor = (q / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    return prefactor * r ** ((q / 2.0 - 1.0) / 2.0) * np.exp(-0.5 * r ** (q / 2.0))


# ── shared settings ───────────────────────────────────────────────────────────

X_LIM = 4.5
x = np.linspace(-X_LIM, X_LIM, 1800)   # even count → no point at x = 0

SLIDE_CURVES = [
    # (q,   legend label,                   color,     linestyle, linewidth)
    (2.0, "Gaussian  (q = 2)",              "#2166ac", "-",       3.0),
    (1.5, "QEP default  (q = 1.5)",         "#e07b39", "--",      2.8),
    (1.0, "QEP  (q = 1.0)",                 "#c0392b", ":",       2.5),
]

BACKUP_CURVES = [
    (2.0, "q = 2  (Gaussian)",              "#2166ac", "-",       2.5),
    (1.8, "q = 1.8",                        "#74add1", "--",      2.0),
    (1.5, "q = 1.5  [project default]",     "#e07b39", "-.",      2.0),
    (1.0, "q = 1.0",                        "#c0392b", ":",       2.0),
]


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — slide-ready (single panel, log-y scale, 3 curves)
# ═════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for q, label, color, ls, lw in SLIDE_CURVES:
    y = qexp_density_1d(x, q=q)
    ax.semilogy(x, y, label=label, color=color, linestyle=ls, linewidth=lw)

# ── y-axis: show from 1e-5 to 0.8 (Gaussian peaks at 0.40; q<2 cusp is above range) ──
ax.set_ylim(1e-5, 0.8)
ax.set_xlim(-X_LIM, X_LIM)
ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

# ── shade tail regions ─────────────────────────────────────────────────────
TAIL_START = 2.3
for lo, hi in [(-X_LIM, -TAIL_START), (TAIL_START, X_LIM)]:
    ax.axvspan(lo, hi, color="#fffacd", alpha=0.6, zorder=0, linewidth=0)

# ── tail annotation ────────────────────────────────────────────────────────
# At x ~ 3.4, Gaussian ~ 2e-4; q=1.0 ~ 0.012  (60x heavier)
ax.annotate(
    "heavier tails\nfor smaller q",
    xy=(3.4, 2e-3),          # tip: somewhere between q=1.0 curve and Gaussian
    xytext=(3.05, 8e-5),     # text sits lower, in the yellow band
    fontsize=11.5,
    color="#8B3A0F",
    style="italic",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#8B3A0F", lw=1.3),
)

# ── subtle central label ──────────────────────────────────────────────────
ax.text(0.0, 0.72, "sharper center",
        ha="center", va="top", fontsize=9, color="#aaaaaa", style="italic",
        transform=ax.get_xaxis_transform())

# ── labels / legend ────────────────────────────────────────────────────────
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("Density p(x) [log scale]", fontsize=14)
ax.set_title("How q Changes Distribution Shape",
             fontsize=15, fontweight="bold", pad=12)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12.5, framealpha=0.92, loc="lower center",
          bbox_to_anchor=(0.5, 0.01))
ax.grid(True, alpha=0.20, which="both", linewidth=0.6)

plt.tight_layout()
fig.savefig("results/gaussian_vs_qep_density_slide.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: results/gaussian_vs_qep_density_slide.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — technical backup (2-panel: linear left, log right, 4 curves)
# ═════════════════════════════════════════════════════════════════════════════

fig2, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor("white")

for q, label, color, ls, lw in BACKUP_CURVES:
    y = qexp_density_1d(x, q=q)
    ax_lin.plot(x,    y, label=label, color=color, linestyle=ls, linewidth=lw)
    ax_log.semilogy(x, y, label=label, color=color, linestyle=ls, linewidth=lw)

# ── linear panel ───────────────────────────────────────────────────────────
ax_lin.set_xlim(-X_LIM, X_LIM)
ax_lin.set_ylim(0, 0.62)
ax_lin.set_xlabel("x", fontsize=13)
ax_lin.set_ylabel("p(x)", fontsize=13)
ax_lin.set_title("Density — linear scale", fontsize=12)
ax_lin.legend(fontsize=10.5, framealpha=0.9)
ax_lin.grid(True, alpha=0.22, linewidth=0.6)
ax_lin.spines["top"].set_visible(False)
ax_lin.spines["right"].set_visible(False)
ax_lin.tick_params(labelsize=11)
ax_lin.text(0.0, 0.605,
            "curves for q < 2 peak above this axis near x = 0",
            ha="center", va="top", fontsize=8, color="#888888", style="italic",
            transform=ax_lin.get_xaxis_transform())

# ── log panel ──────────────────────────────────────────────────────────────
ax_log.set_xlim(-X_LIM, X_LIM)
ax_log.set_ylim(1e-6, 0.8)
ax_log.set_xlabel("x", fontsize=13)
ax_log.set_ylabel("p(x)   [log scale]", fontsize=13)
ax_log.set_title("Density — log scale  (shows tail behaviour)", fontsize=12)
ax_log.legend(fontsize=10.5, framealpha=0.9)
ax_log.grid(True, alpha=0.22, linewidth=0.6, which="both")
ax_log.spines["top"].set_visible(False)
ax_log.spines["right"].set_visible(False)
ax_log.tick_params(labelsize=11)
for lo, hi in [(-X_LIM, -TAIL_START), (TAIL_START, X_LIM)]:
    ax_log.axvspan(lo, hi, color="#fffacd", alpha=0.55, zorder=0, linewidth=0)
ax_log.text(3.3, 2e-5, "heavier\ntails", ha="center", fontsize=9.5,
            color="#8B3A0F", style="italic")

fig2.suptitle(
    "Q-Exponential Distribution Family  (mu=0, sigma=1)\n"
    "q=2 -> Gaussian; q<2 -> heavier tails",
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig2.savefig("results/gaussian_vs_qep_density.png", dpi=180, bbox_inches="tight")
fig2.savefig("results/gaussian_vs_qep_density.pdf",            bbox_inches="tight")
plt.close(fig2)
print("Saved: results/gaussian_vs_qep_density.png")
print("Saved: results/gaussian_vs_qep_density.pdf")
print("Done.")
