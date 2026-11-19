"""Exact Mellin representation of the covariance-matched qED, numerical 1-D integration."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import digamma, polygamma, gammaln, loggamma, logsumexp, roots_hermitenorm
from scipy.optimize import brentq

@lru_cache(None)
def hermite(order):
    return roots_hermitenorm(order)

class ScalePrior:
    """Spline of log prior density for log S via exponential tilting and Fourier inversion.

    R_chi^2 ~ Gamma(k,2). QEP radius squared has law (R_chi^2)^b/a.
    S independently multiplies a Gaussian covariance. On log scale its cumulant
    is K(z)=z*c+logGamma(k+b*z)-logGamma(k+z). This gives the target radial
    characteristic function upon multiplication with an independent chi square.
    The difference of log-gamma Levy measures is nonnegative for b>=1, giving
    existence. Exponential tilting avoids catastrophic tail cancellation.
    """
    def __init__(self,n,q=1.5):
        k=n/2
        b=2/q
        self.k,self.b=k,b
        self.c=c=np.log(k)+gammaln(k)-gammaln(k+b)
        self.sd=np.sqrt((b*b-1)*polygamma(1,k))
        # Denser than a tenth of a log-scale SD in the central working region.
        u=np.linspace(-1,1,1001)
        z=k*np.where(u<0,0.70/b,3.0)*np.sinh(5*u)/np.sinh(5)
        x=c+b*digamma(k+b*z)-digamma(k+z)
        sd=np.sqrt(b*b*polygamma(1,k+b*z)-polygamma(1,k+z))
        t=np.linspace(-11,11,177)[None,:]/sd[:,None]
        a1=k+b*z[:,None]; a2=k+z[:,None]
        phase=(loggamma(a1+1j*b*t)-gammaln(a1)
               -loggamma(a2+1j*t)+gammaln(a2)+1j*t*(c-x[:,None]))
        integral=np.trapz(np.exp(phase).real,dx=22/176,axis=1)/(2*np.pi*sd)
        if not (integral>0).all(): raise ArithmeticError("nonpositive tilted density")
        y=z*c+gammaln(k+b*z)-gammaln(k+z)-z*x+np.log(integral)
        self.spline=CubicSpline(x,y)
        self.bounds=(x.min(),x.max())
    def logpdf(self,x,derivative=0):
        if np.any(np.asarray(x)<self.bounds[0]) or np.any(np.asarray(x)>self.bounds[1]):
            raise ArithmeticError("quadrature outside verified prior grid")
        return self.spline(x,derivative)

def evaluate(lam, residual, s, nu, prior, order=12, predict=False):
    """Marginal likelihood and posterior spectral shrinkage, with adaptive Gaussian quadrature."""
    energy=residual**2/s
    n=len(lam)
    def at(x,derivative=0):
        signal=lam*np.exp(x)
        var=signal+nu
        f=signal/var
        chisq=energy/var
        if derivative==1: return .5*np.sum(f*(chisq-1))+prior.logpdf(x,1)
        if derivative==2: return .5*np.sum(f*(1-f)*(chisq-1)-f*f*chisq)+prior.logpdf(x,2)
        return -.5*(n*np.log(2*np.pi*s)+np.log(var).sum()+chisq.sum())+prior.logpdf(x)
    if prior is None:
        f=lam/(lam+nu)
        return -.5*(n*np.log(2*np.pi*s)+np.log(lam+nu).sum()+(energy/(lam+nu)).sum()), f if predict else None
    lo,hi=prior.bounds
    left,right=at(lo,1),at(hi,1)
    if not left>0>right: raise ArithmeticError("posterior mode not bracketed")
    mode=brentq(lambda x:at(x,1),lo,hi,xtol=2e-11)
    width=(-at(mode,2))**-.5
    nodes,weights=hermite(order)
    xx=mode+width*nodes
    logs=np.array([at(x) for x in xx])+nodes**2/2+np.log(weights*width)
    ll=logsumexp(logs)
    if predict:
        ratios=np.zeros_like(lam)
        for x,w in zip(xx,np.exp(logs-ll)):
            ratios+=w*lam/(lam+nu*np.exp(-x))
        return ll,ratios
    return ll,None
