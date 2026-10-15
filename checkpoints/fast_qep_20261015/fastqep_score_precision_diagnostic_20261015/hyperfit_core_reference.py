"""Evidence scores for the declared finite-tile QEP + Gaussian noise model.

theta=(m, log_s, log_nu, log_beta_rows, log_beta_cols); tau2=s*nu.
Uses the byte-unchanged posterior/sampler. No new latent target or approximation.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from py_core.fast_qep_prototype import KroneckerGrid, SpectralPosterior

NAMES = ['mean', 'log_signal_variance', 'log_noise_ratio', 'log_beta_rows', 'log_beta_cols']


def kernel_derivative(size, beta):
    x = np.linspace(0, 1, size)
    t = np.sqrt(5)*beta*np.abs(x[:, None]-x)
    return -(t*t+t*t*t)*np.exp(-t)/3


class ScoreGeometry:
    def __init__(self, grid):
        self.grid = grid
        nr, nc = grid.shape
        # Both correlation factors have unit diagonal; recover their eigenvalues
        # from the existing products, avoiding a second eigenvector convention.
        lam = grid.eigenvalues.reshape(grid.shape, order='F')
        er, ec = lam.sum(axis=1)/nc, lam.sum(axis=0)/nr
        self.condition = {'row_eigen_min':float(er.min()),'row_eigen_max':float(er.max()),
                          'col_eigen_min':float(ec.min()),'col_eigen_max':float(ec.max()),
                          'row_condition':float(er.max()/er.min()),
                          'col_condition':float(ec.max()/ec.min())}
        dr = grid.row_vectors.T@kernel_derivative(nr, grid.beta_rows)@grid.row_vectors
        dc = grid.col_vectors.T@kernel_derivative(nc, grid.beta_cols)@grid.col_vectors
        self.br = dr/np.sqrt(er[:, None]*er)
        self.bc = dc/np.sqrt(ec[:, None]*ec)
        self.diagonals = [np.tile(np.diag(self.br), nc), np.repeat(np.diag(self.bc), nr)]
        self.traces = [float(d.sum()) for d in self.diagonals]
        self.ones = grid.forward(np.ones(grid.shape))

    def apply(self, z, axis):
        z = np.asarray(z)
        nr, nc = self.grid.shape
        flat = z.reshape(-1, self.grid.n)
        mats = flat.reshape(-1, nc, nr).transpose(0, 2, 1)
        result = self.br@mats if axis == 0 else mats@self.bc
        return result.transpose(0, 2, 1).reshape(z.shape)


def state(tile, theta, q):
    grid = KroneckerGrid.matern52(tile.shape, *np.exp(theta[3:5]))
    observations = grid.forward(tile-theta[0])
    target = SpectralPosterior(grid.eigenvalues, observations, np.exp(theta[1]),
                               np.exp(theta[1]+theta[2]), q)
    return grid, target, ScoreGeometry(grid)


class EvidenceScores:
    """Raw Fisher scores and exact-identity Stein control variates.

    Controls have zero expectation under the CURRENT white posterior. Their
    coefficients are fixed from the analytic q=2 reference before sampling.
    Corrected scores are estimators, not latent draws or nonlinear moments.
    """
    def __init__(self, target, geometry):
        self.target, self.geo = target, geometry
        self.mu = target.gaussian_white_mean()
        self.precision = target.mass_diag
        self.m_control = -geometry.ones*target.gaussian_root/target.noise_variance/self.precision
        self.d_mu = [geometry.apply(self.mu, axis) for axis in (0, 1)]

    def __call__(self, z):
        z = np.asarray(z)
        t, g = self.target, self.geo
        h = t.latent(z)
        e = t.observations-h
        rho2 = np.sum(z*z, axis=-1)
        if np.any(rho2 <= 0):
            raise ValueError('Zero radius is excluded from the physical prior score.')
        noise = -t.n/2+np.sum(e*e, axis=-1)/(2*t.noise_variance)
        raw = [np.sum(e*g.ones, axis=-1)/t.noise_variance,
               t.q/4*(rho2-t.n)+noise, noise]
        dz = [g.apply(z, axis) for axis in (0, 1)]
        coeff = t.q/4-t.n*(t.q-2)/(4*rho2)
        raw.extend(-0.5*g.traces[axis]+coeff*np.sum(z*dz[axis], axis=-1) for axis in (0, 1))
        raw = np.stack(raw, axis=-1)
        _, grad_u = t.potential_gradient(z)
        centered = z-self.mu
        controls = [np.sum(self.m_control*grad_u, axis=-1),
                    0.5*np.sum(centered*grad_u, axis=-1)-t.n/2]
        fnu = 0.5*(1-1/self.precision)*centered-self.mu/self.precision
        controls.append(np.sum(fnu*grad_u, axis=-1)-0.5*np.sum(1-1/self.precision))
        for axis in (0, 1):
            field = 0.5*(dz[axis]+self.d_mu[axis])/self.precision
            trace = 0.5*np.sum(g.diagonals[axis]/self.precision)
            controls.append(np.sum(field*grad_u, axis=-1)-trace)
        controls = np.stack(controls, axis=-1)
        return raw, raw-controls


def gaussian_evidence(tile, theta, grid=None, geometry=None):
    if grid is None:
        grid, _, geometry = state(tile, theta, 2.)
    if geometry is None:
        geometry = ScoreGeometry(grid)
    s, nu = np.exp(theta[1:3])
    residual = grid.forward(tile-theta[0])
    denom = s*(grid.eigenvalues+nu)
    inv_residual = residual/denom
    ll = -0.5*np.sum(np.log(2*np.pi*denom)+residual*inv_residual)
    score = [geometry.ones@inv_residual, -grid.n/2+0.5*(residual@inv_residual),
             0.5*np.sum(s*nu*inv_residual**2-nu/(grid.eigenvalues+nu))]
    whitened = np.sqrt(grid.eigenvalues)*inv_residual
    for axis in (0, 1):
        quad = s*whitened@geometry.apply(whitened, axis)
        trace = np.sum(grid.eigenvalues/(grid.eigenvalues+nu)*geometry.diagonals[axis])
        score.append(0.5*(quad-trace))
    return float(ll), np.asarray(score)


def nuisance_update(grid, target, tile, latent_mean, latent_norm2_mean, quadratic_power_mean):
    """One conditional ECM proposal. Input moments must come from RAW latents.

    Integration can produce these moments exactly; MCMC cannot guarantee ascent.
    This routine does not authorize or accept a noisy optimization step.
    """
    n = grid.n
    xx = grid.forward(np.ones(grid.shape))
    m = float(np.mean(tile)-xx@latent_mean/n)
    residual = grid.forward(tile-m)
    b = float(residual@residual-2*residual@latent_mean+latent_norm2_mean)
    if not np.isfinite(b) or b <= 0 or quadratic_power_mean <= 0:
        raise ValueError('Invalid raw posterior moments for the nuisance update.')
    nu = target.noise_variance/target.signal_variance
    def f(log_s):
        return (target.q/4*np.exp(target.q/2*(np.log(target.a)-log_s))*quadratic_power_mean
                +b/(2*np.exp(log_s)*nu)-n*(target.q+2)/4)
    old = np.log(target.signal_variance)
    logs = brentq(f, old-20, old+20, xtol=1e-12)
    lognu = np.log(b/n)-logs
    return np.array([m, logs, lognu, np.log(grid.beta_rows), np.log(grid.beta_cols)])
