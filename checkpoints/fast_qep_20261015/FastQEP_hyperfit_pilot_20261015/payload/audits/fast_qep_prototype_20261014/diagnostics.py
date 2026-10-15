"""Small NumPy/SciPy implementation of split rank R-hat and Geyer ESS.

References: Vehtari et al. (2021), arXiv:1903.08008; Stan reference manual,
Posterior Analysis. ESS is conservatively capped at the number of draws.
MCSE uses untransformed values, not rank-normalized posterior variance.
"""

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.special import ndtri
from scipy.stats import rankdata


def split_chains(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite (chains, draws) with at least two chains and eight draws.")
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def rank_normalize(x):
    ranks = rankdata(x.ravel(), method="average")
    return ndtri((ranks - 0.375) / (ranks.size + 0.25)).reshape(x.shape)


def basic_rhat(x):
    _, n = x.shape
    within = float(np.var(x, axis=1, ddof=1).mean())
    between_mean = float(np.var(x.mean(axis=1), ddof=1))
    if within == 0:
        return 1.0 if between_mean == 0 else float("inf")
    return float(np.sqrt(((n - 1) / n * within + between_mean) / within))


def geyer_ess(x):
    """Initial positive/monotone paired autocorrelations, including lag zero."""
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    nfft = next_fast_len(2 * n)
    ft = rfft(centered, n=nfft, axis=1)
    acov = irfft(ft * ft.conj(), n=nfft, axis=1)[:, :n] / n
    within = float(acov[:, 0].mean() * n / (n - 1))
    var_plus = (n - 1) / n * within + float(np.var(x.mean(axis=1), ddof=1))
    if var_plus <= 0:
        return float(m * n), 0.0
    rho = 1 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[:2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    pairs = pairs[:int(stop[0])] if stop.size else pairs
    tau = max(1.0, -1 + 2 * np.minimum.accumulate(pairs).sum()) if pairs.size else 1.0
    return float(min(m * n, m * n / tau)), float(var_plus)


def summarize(x):
    split = split_chains(x)
    normalized = rank_normalize(split)
    folded = rank_normalize(abs(split - np.median(split)))
    rhat = max(basic_rhat(normalized), basic_rhat(folded))
    ess_bulk, _ = geyer_ess(normalized)
    quantiles = np.quantile(split, [0.05, 0.95])
    ess_tail = min(geyer_ess((split <= cut).astype(float))[0] for cut in quantiles)
    ess_mean, var_plus = geyer_ess(split)
    spectral_mcse = np.sqrt(var_plus / ess_mean)
    m, n = x.shape
    batch_errors = []
    for block in (100, 500):
        count = n // block
        if count >= 2:
            batches = x[:, :count * block].reshape(m, count, block).mean(axis=2)
            batch_errors.append(float(np.sqrt(np.var(batches, axis=1, ddof=1).mean() / (m * count))))
    chain_mcse = float(np.std(x.mean(axis=1), ddof=1) / np.sqrt(m))
    return {
        "mean": float(x.mean()),
        "sd": float(np.std(x, ddof=1)),
        "rhat_rank_folded_split": float(rhat),
        "ess_bulk": float(ess_bulk),
        "ess_tail": float(ess_tail),
        "ess_mean_conservative": float(ess_mean),
        "mcse_spectral": float(spectral_mcse),
        "mcse_batch_max": max(batch_errors, default=0.0),
        "mcse_between_chains": chain_mcse,
        "mcse": max(float(spectral_mcse), max(batch_errors, default=0.0), chain_mcse),
    }


def self_check():
    rng = np.random.default_rng(104713)
    iid = rng.normal(size=(4, 20000))
    iid_result = summarize(iid)
    ar = np.empty_like(iid)
    ar[:, 0] = iid[:, 0]
    for j in range(1, ar.shape[1]):
        ar[:, j] = 0.8 * ar[:, j - 1] + 0.6 * iid[:, j]
    ar_result = summarize(ar)
    shifted = iid + np.arange(4)[:, None] * 1.5
    shifted_result = summarize(shifted)
    assert iid_result["rhat_rank_folded_split"] < 1.01
    assert iid_result["ess_bulk"] > 0.8 * iid.size
    expected_ar_ess = ar.size * (1 - 0.8) / (1 + 0.8)
    assert 0.7 < ar_result["ess_mean_conservative"] / expected_ar_ess < 1.3
    assert shifted_result["rhat_rank_folded_split"] > 1.1
    assert shifted_result["ess_bulk"] < 400
    return {"iid": iid_result, "ar1": ar_result, "ar1_asymptotic_ess": expected_ar_ess,
            "shifted_chains": shifted_result}
