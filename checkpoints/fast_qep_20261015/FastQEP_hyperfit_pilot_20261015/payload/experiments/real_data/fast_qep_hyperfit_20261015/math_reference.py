#!/usr/bin/env python3
"""Standalone Fast-QEP hyperparameter algebra audit; NOT a cell experiment.

Model: y=m*1+v+e, Cov(v)=s*K, v~qED_n(0,s*K/a(n,q)),
e~N(0,s*nu*I), independent. Coordinates: m, log(s), log(nu), log(beta...)
Requires numpy/scipy only. No R, microscopy data, MCMC or segmentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import gammaln


def covariance_factor(n, q):
    if q == 2:
        return 1.0
    return float(np.exp((2/q)*np.log(2)+gammaln(n/2+2/q)
                        -gammaln(n/2)-np.log(n)))


def matern(x, log_beta):
    """R parameterization, and derivative wrt log inverse range."""
    d = np.abs(x[:, None]-x[None, :])
    t = np.sqrt(5)*np.exp(log_beta)*d
    et = np.exp(-t)
    return (1+t+t*t/3)*et, -(t*t+t*t*t)*et/3


def kernel(theta, shape):
    if len(shape) == 1:
        k, dk = matern(np.linspace(0, 1, shape[0]), theta[3])
        return k, [dk]
    nr, nc = shape
    kr, dr = matern(np.linspace(0, 1, nr), theta[3])
    kc, dc = matern(np.linspace(0, 1, nc), theta[4])
    return np.kron(kc, kr), [np.kron(kc, dr), np.kron(dc, kr)]


def log_joint_and_score(v, y, theta, q, shape):
    """Score holds the physical centered latent v fixed, not whitened z."""
    n = len(y)
    m, s, nu = theta[0], np.exp(theta[1]), np.exp(theta[2])
    tau2 = s*nu
    k, derivatives = kernel(theta, shape)
    sign, ld = np.linalg.slogdet(k)
    if sign <= 0:
        raise ValueError('K must be positive definite; no hidden eigenvalue clipping')
    w = np.linalg.solve(k, v)
    quad = float(v@w)
    if quad <= 0:
        raise ValueError('v=0 has a singular q<2 density and is excluded')
    a = covariance_factor(n, q)
    r = a*quad/s
    alpha = n*(q-2)/4
    rp = r**(q/2)
    lp = np.log(q/2)-n/2*np.log(2*np.pi)-0.5*(n*np.log(s/a)+ld)
    lp += alpha*np.log(r)-0.5*rp
    e = y-m-v
    ll = -n/2*np.log(2*np.pi*tau2)-float(e@e)/(2*tau2)
    gt = -n/2+float(e@e)/(2*tau2)
    score = [e.sum()/tau2, q/4*(rp-n)+gt, gt]
    for dk in derivatives:
        score.append(-0.5*np.trace(np.linalg.solve(k, dk))
                     +(q/4*rp-alpha)*(w@dk@w)/quad)
    return float(lp+ll), np.array(score)


def kron_score(v, y, theta, q, shape):
    """Same complete-data score without an n-by-n covariance matrix."""
    nr, nc = shape
    n = nr*nc
    kr, dr = matern(np.linspace(0, 1, nr), theta[3])
    kc, dc = matern(np.linspace(0, 1, nc), theta[4])
    lr, ur = np.linalg.eigh(kr)
    lc, uc = np.linalg.eigh(kc)
    if min(lr.min(), lc.min()) <= 0:
        raise ValueError('Non-positive eigenvalue; no hidden clipping')
    z = ur.T@v.reshape(shape, order='F')@uc
    ainv = z/(lr[:, None]*lc[None, :])
    quad = np.sum(z*ainv)
    br, bc = ur.T@dr@ur, uc.T@dc@uc
    row_num = np.sum(ainv*(br@ainv)*lc[None, :])
    col_num = np.sum(ainv*(ainv@bc)*lr[:, None])
    traces = [nc*np.sum(np.diag(br)/lr), nr*np.sum(np.diag(bc)/lc)]
    m, s, nu = theta[0], np.exp(theta[1]), np.exp(theta[2])
    rp = (covariance_factor(n, q)*quad/s)**(q/2)
    e = y-m-v
    gt = -n/2+np.dot(e, e)/(2*s*nu)
    alpha = n*(q-2)/4
    out = [e.sum()/(s*nu), q/4*(rp-n)+gt, gt]
    out += [-0.5*tr+(q/4*rp-alpha)*val/quad
            for tr, val in zip(traces, [row_num, col_num])]
    return np.array(out)


def finite_difference(fn, x, h=1e-5):
    eye = np.eye(len(x))*h
    return np.array([(fn(x+u)-fn(x-u))/(2*h) for u in eye])


def gaussian_evidence(y, theta, shape):
    n = len(y)
    k, dks = kernel(theta, shape)
    s, nu = np.exp(theta[1]), np.exp(theta[2])
    sig = s*(k+nu*np.eye(n))
    inv = np.linalg.inv(sig)
    e = y-theta[0]
    w = inv@e
    ll = -0.5*(n*np.log(2*np.pi)+np.linalg.slogdet(sig)[1]+e@w)
    ds = [sig, s*nu*np.eye(n)]+[s*dk for dk in dks]
    score = np.array([w.sum()]+[0.5*(w@dk@w-np.trace(inv@dk)) for dk in ds])
    mean = s*k@w
    cov = s*k-s*s*k@inv@k
    return float(ll), score, mean, cov


def quadrature(y, theta, q, angles=128):
    """n=2 only: t=R**q/2~Exp(1), direction uniform on circle.

    Angular trapezoid plus adaptive radial quadrature. Returns exact-integral
    approximations, not samples. Used as an independent evidence/score oracle.
    """
    assert len(y) == 2 and len(theta) == 4
    n = 2
    m, s, nu = theta[0], np.exp(theta[1]), np.exp(theta[2])
    tau2 = s*nu
    k, dks = kernel(theta, (2,))
    ki = np.linalg.inv(k)
    chol = np.linalg.cholesky(k)
    a = covariance_factor(n, q)
    alpha = n*(q-2)/4
    phi = (np.arange(angles)+0.5)*(2*np.pi/angles)
    direction = np.column_stack([np.cos(phi), np.sin(phi)])@chol.T
    kd = direction@ki
    ratio = np.sum((kd@dks[0])*kd, axis=1)
    tr = np.trace(ki@dks[0])

    def integrand(t):
        radius = (2*t)**(1/q)
        v = np.sqrt(s/a)*radius*direction
        e = y[None, :]-m-v
        norm2 = np.sum(e*e, axis=1)
        weight = np.exp(-t-norm2/(2*tau2))/(2*np.pi*tau2)
        gt = -1+norm2/(2*tau2)
        gs = q/4*(2*t-n)+gt
        gb = -0.5*tr+(q/4*2*t-alpha)*ratio
        score = np.column_stack([e.sum(axis=1)/tau2, gs, gt, gb])
        # At current K: (v^T K^-1 v)^(q/2)=(s/a)^(q/2)*2t.
        qp = (s/a)**(q/2)*2*t
        moments = np.column_stack([np.ones(angles), score, v, v*v,
                                   np.full(angles, qp)])
        return np.mean(weight[:, None]*moments, axis=0)

    integral, error = quad_vec(integrand, 0, np.inf, epsabs=2e-11,
                               epsrel=2e-10, limit=400)
    norm = integral[0]
    out = integral/norm
    return {'log_evidence': float(np.log(norm)), 'score': out[1:5],
            'latent_mean': out[5:7], 'latent_second': out[7:9],
            'radial_moment': float(out[9]),
            'integral_error_bound': float(error)}


def ecm_nuisance_update(y, theta, q, posterior):
    """One exact-expectation ECM sweep for m,s,nu with K fixed.

    For real MCMC a separate error-aware acceptance rule is required.
    Never replace E[v*v] by E[v]**2 or use mean-control-variate samples here.
    """
    n = len(y)
    nxt = theta.copy()
    ev, ev2 = posterior['latent_mean'], posterior['latent_second']
    nxt[0] = np.mean(y-ev)
    err2 = float(np.sum((y-nxt[0])**2-2*(y-nxt[0])*ev+ev2))
    a = covariance_factor(n, q)
    nu = np.exp(theta[2])
    moment = posterior['radial_moment']

    def root(logs):
        return (q/4*np.exp(q/2*(np.log(a)-logs))*moment
                +err2/(2*np.exp(logs)*nu)-n*(q+2)/4)

    nxt[1] = brentq(root, theta[1]-20, theta[1]+20, xtol=1e-13)
    nxt[2] = np.log(err2/(n*np.exp(nxt[1])))
    return nxt


def audit():
    t0 = time.perf_counter()
    results = {'purpose': 'numerical validation, not a segmentation experiment',
               'model': 'finite-tile covariance-matched qED + independent Gaussian noise',
               'coordinates': ['m', 'log_s', 'log_nu', 'log_beta_row', 'log_beta_col'],
               'seed': 20261015, 'checks': {}}
    checks = results['checks']
    rng = np.random.default_rng(20261015)
    shape = (3, 4)
    theta = np.array([0.15, np.log(1.2), np.log(0.45), np.log(1.7), np.log(2.3)])
    y, v = rng.normal(size=12), rng.normal(size=12)
    for q in (2.0, 1.5):
        val, score = log_joint_and_score(v, y, theta, q, shape)
        fd = finite_difference(lambda th: log_joint_and_score(v, y, th, q, shape)[0], theta)
        ks = kron_score(v, y, theta, q, shape)
        checks[f'complete_score_q{q}'] = {'max_abs_error': float(np.max(np.abs(score-fd))),
                                          'tolerance': 1e-6}
        checks[f'kronecker_score_q{q}'] = {'max_abs_error': float(np.max(np.abs(score-ks))),
                                           'tolerance': 1e-8}
        if q == 2:
            k, _ = kernel(theta, shape)
            s, nu = np.exp(theta[1]), np.exp(theta[2])
            gp_prior = -0.5*(12*np.log(2*np.pi)+np.linalg.slogdet(s*k)[1]+v@np.linalg.solve(s*k,v))
            e = y-theta[0]-v
            gp_lik = -0.5*(12*np.log(2*np.pi*s*nu)+e@e/(s*nu))
            checks['q2_joint_density'] = {'max_abs_error': abs(val-gp_prior-gp_lik), 'tolerance': 1e-10}

    # Profile GP mean/scale in exactly the parameterization used by original R.
    k, _ = kernel(theta, shape)
    amat = k+np.exp(theta[2])*np.eye(12)
    oi = np.linalg.solve(amat, np.ones(12))
    thp = theta.copy()
    thp[0] = (oi@y)/oi.sum()
    e = y-thp[0]
    ss = e@np.linalg.solve(amat, e)
    thp[1] = np.log(ss/12)
    ll, g, _, _ = gaussian_evidence(y, thp, shape)
    # R drops constants including -n/2*log(n); compare after restoring them.
    r_obj = 0.5*np.linalg.slogdet(amat)[1]+6*np.log(ss)
    restored = r_obj+6*(np.log(2*np.pi)+1-np.log(12))
    checks['q2_R_profile_algebra'] = {'max_abs_error': float(max(abs(ll+restored),abs(g[0]),abs(g[1]))),
                                      'tolerance': 1e-10,
                                      'note': 'R formula checked; R not executed in this review'}

    # Independent low-dimensional integration, fixed before results inspected.
    y2 = np.array([0.7, -0.4])
    th2 = np.array([0.1, np.log(1.3), np.log(0.4/1.3), np.log(1.4)])
    oracle = {}
    for q in (2.0, 1.5):
        post = quadrature(y2, th2, q)
        refined = quadrature(y2, th2, q, angles=256)
        fd = finite_difference(lambda th: quadrature(y2, th, q)['log_evidence'], th2)
        checks[f'evidence_fisher_score_q{q}'] = {'max_abs_error': float(np.max(np.abs(post['score']-fd))),
                                                'tolerance': 2e-6}
        checks[f'angular_refinement_q{q}'] = {'max_abs_error': float(max(
            abs(post['log_evidence']-refined['log_evidence']),
            np.max(np.abs(post['score']-refined['score'])))), 'tolerance': 1e-8}
        if q == 2:
            ll, g, mu, cov = gaussian_evidence(y2, th2, (2,))
            err = max(abs(post['log_evidence']-ll), np.max(abs(post['score']-g)),
                      np.max(abs(post['latent_mean']-mu)),
                      np.max(abs(post['latent_second']-(mu*mu+np.diag(cov)))))
            checks['q2_integral_vs_analytic_GP'] = {'max_abs_error': float(err), 'tolerance': 1e-8}
        # One evidence ascent step updates every coordinate, including beta.
        step = 0.1/max(1.0, np.linalg.norm(post['score']))
        candidate = th2+step*post['score']
        llnew = quadrature(y2, candidate, q)['log_evidence']
        checks[f'evidence_ascent_q{q}'] = {'increase': float(llnew-post['log_evidence']),
                                          'minimum': 0.0}
        # Five deterministic ECM sweeps validate the nuisance updating formula.
        current = th2.copy()
        trajectory = [post['log_evidence']]
        for _ in range(5):
            pp = quadrature(y2, current, q)
            current = ecm_nuisance_update(y2, current, q, pp)
            trajectory.append(quadrature(y2, current, q)['log_evidence'])
        checks[f'ECM_monotonic_q{q}'] = {'min_increase': float(np.min(np.diff(trajectory))),
                                        'minimum': -1e-8}
        oracle[str(q)] = {'initial_theta': th2.tolist(), 'initial_score': post['score'].tolist(),
                         'initial_log_evidence': post['log_evidence'],
                         'joint_step_theta': candidate.tolist(), 'joint_step_log_evidence': llnew,
                         'nuisance_ECM_log_evidence': trajectory,
                         'nuisance_ECM_final_theta': current.tolist(),
                         'note': 'Two-observation identity check; not an identifiable fitted-image benchmark.'}
    results['quadrature_fixture'] = {'y': y2.tolist(), 'arms': oracle}
    results['passed'] = True
    for check in checks.values():
        if 'max_abs_error' in check:
            check['passed'] = bool(check['max_abs_error'] < check['tolerance'])
        elif 'increase' in check:
            check['passed'] = bool(check['increase'] > check['minimum'])
        else:
            check['passed'] = bool(check['min_increase'] >= check['minimum'])
        results['passed'] &= check['passed']
    results['runtime_s'] = time.perf_counter()-t0
    results['environment'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__}
    results['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation.json'))
    args = parser.parse_args()
    out = audit()
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    for key, check in out['checks'].items():
        print(key, 'PASS' if check['passed'] else 'FAIL', check)
    print('PASS' if out['passed'] else 'FAIL', args.output)
    raise SystemExit(0 if out['passed'] else 1)
