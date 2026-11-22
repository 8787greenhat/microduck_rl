"""Bounded PPO actions with matched sampling, likelihood, KL and ONNX paths."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import Distribution


class _DeterministicSquash(nn.Module):
    def __init__(self,mean_limit=2.0):
        super().__init__();self.mean_limit=mean_limit

    def forward(self,x):
        return torch.tanh(self.mean_limit*torch.tanh(x/self.mean_limit))


class BoundedSquashedGaussian(Distribution):
    """Tanh-normal in normalized action coordinates, with bounded latent std.

    The common invertible tanh transform leaves KL equal to latent Normal KL.
    Entropy uses analytic Normal entropy plus a reparameterized Jacobian sample;
    it therefore regularizes the action actually used by the environment.
    """
    def __init__(self,output_dim,init_std=.4,min_std=.05,max_std=.6,mean_limit=2.):
        super().__init__(output_dim)
        if not 0<min_std<init_std<max_std or mean_limit<=0:
            raise ValueError('Require 0 < min_std < init_std < max_std and positive mean_limit')
        self.min_std=min_std;self.max_std=max_std;self.mean_limit=mean_limit
        fraction=(init_std-min_std)/(max_std-min_std)
        self.std_logits=nn.Parameter(torch.full((output_dim,),math.log(fraction/(1-fraction))))
        self._distribution=None

    def latent_mean(self,x):return self.mean_limit*torch.tanh(x/self.mean_limit)
    def effective_std(self):return self.min_std+(self.max_std-self.min_std)*torch.sigmoid(self.std_logits)
    def update(self,mlp_output):
        mean=self.latent_mean(mlp_output)
        self._distribution=Normal(mean,self.effective_std().expand_as(mean),validate_args=False)
    def sample(self):return torch.tanh(self._distribution.sample())
    def deterministic_output(self,mlp_output):return torch.tanh(self.latent_mean(mlp_output))
    def as_deterministic_output_module(self):return _DeterministicSquash(self.mean_limit)
    @property
    def input_dim(self):return self.output_dim
    @property
    def mean(self):return torch.tanh(self._distribution.mean)
    @property
    def std(self):return self._distribution.stddev  # logged latent units, bounded
    @staticmethod
    def log_jacobian(z):return 2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
    @property
    def entropy(self):
        z=self._distribution.rsample()
        return (self._distribution.entropy()+self.log_jacobian(z)).sum(-1)
    @property
    def params(self):return self._distribution.mean,self._distribution.stddev
    def log_prob(self,outputs):
        values=outputs.clamp(-1+1e-6,1-1e-6)
        z=torch.atanh(values)
        return (self._distribution.log_prob(z)-self.log_jacobian(z)).sum(-1)
    def kl_divergence(self,old_params,new_params):
        return torch.distributions.kl_divergence(Normal(*old_params),Normal(*new_params)).sum(-1)
    def init_mlp_weights(self,mlp):
        linear=[module for module in mlp.modules() if isinstance(module,nn.Linear)][-1]
        nn.init.orthogonal_(linear.weight,gain=.01);nn.init.zeros_(linear.bias)
