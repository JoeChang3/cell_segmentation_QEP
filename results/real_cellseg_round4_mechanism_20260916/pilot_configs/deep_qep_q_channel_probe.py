import sys; sys.path.insert(0,'/Users/zchan/eclipse-workspace/Cell_Seg_QEP')
import numpy as np, torch, gpytorch, qpytorch
torch.set_default_dtype(torch.float64)
from qpytorch.models.deep_qeps import DeepQEP, DeepQEPLayer
from qpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from qpytorch.distributions import MultivariateQExponential
from qpytorch.likelihoods import QExponentialLikelihood
import qpytorch.kernels as K, qpytorch.means as M

class Layer(DeepQEPLayer):
    def __init__(s, ind, outd, power, n_ind=32):
        bshape = torch.Size([]) if outd is None else torch.Size([outd])
        ip = torch.randn(*([outd] if outd else []), n_ind, ind)
        vd = CholeskyVariationalDistribution(n_ind, batch_shape=bshape, power=power)
        vs = VariationalStrategy(s, ip, vd, learn_inducing_locations=True)
        super().__init__(vs, ind, outd)
        s.power = power
        s.mean_module = M.ConstantMean(batch_shape=bshape)
        s.covar_module = K.ScaleKernel(K.MaternKernel(nu=2.5, ard_num_dims=ind,
                                       batch_shape=bshape), batch_shape=bshape)
    def forward(s, x):
        return MultivariateQExponential(s.mean_module(x), s.covar_module(x), power=s.power)

class DQEP(DeepQEP):
    def __init__(s, power, hidden=2):
        super().__init__()
        s.l1 = Layer(2, hidden, power); s.l2 = Layer(hidden, None, power)
        s.likelihood = QExponentialLikelihood(power=power)
    def forward(s, x): return s.l2(s.l1(x))

n=14; g=torch.linspace(0,1,n); a,b=torch.meshgrid(g,g,indexing='ij')
X=torch.stack([a.reshape(-1),b.reshape(-1)],-1).contiguous()

print("CLAIM: in a DEEP QEP (variational, sampled through layers), is the predictive")
print("mean q-dependent at FIXED, identically initialized parameters?")
print("ref: Deep Q-Exponential Processes, Chang/Obite/Zhou/Lan, PMLR v289\n")
res={}
for q in [2.0, 1.5, 1.2]:
    torch.manual_seed(123)                 # identical parameter init per q
    m = DQEP(torch.tensor(float(q))); m.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
        torch.manual_seed(999)             # identical RNG stream per q
        mu = m(X).mean.detach().numpy().ravel()[:5]
    res[q]=mu
    print(f"  q={q}: predictive mean first5 = {np.round(mu,6)}")
base=res[2.0]; print()
for q,v in res.items():
    print(f"  max|mean(q={q}) - mean(q=2)| = {np.abs(v-base).max():.3e}")
print("\n  identical init and identical RNG stream, so any nonzero difference is")
print("  attributable to q entering the layer distributions.")
