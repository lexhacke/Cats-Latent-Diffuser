import numpy as np
import torch
from IPython.display import clear_output

class NoiseSchedule:
  """
  Handles:
  - DDIM inference (with a ddim_mod to skip steps)
  - DDPM inference
  - Forward Noising
  - Linear beta schedule
  - Classifier Free Guidance (w is a hyperparameter for cfg schedule)
  """
  def __init__(self, T, std=1, shape=(4, 64, 64), ddim_mod=10, trainer_mode=False):
    self.T = T
    self.std = std
    self.ddim_mod = ddim_mod
    self.beta = torch.tensor(np.linspace(1e-4, 0.02, T), dtype=torch.float32, device='cpu' if trainer_mode else 'cuda')
    self.alpha = 1 - self.beta
    self.alpha_bar = self.alpha.cumprod(dim=0)
    self.w = torch.full((T,), 7.5, device='cpu' if trainer_mode else 'cuda')
    self.shape = shape

  def noise(self, x, t):
    eps = torch.randn_like(x) * self.std
    return (self.alpha_bar[t]**0.5) * x + ((1-self.alpha_bar[t])**0.5) * eps, eps

  def ddim_step(self, xt, t, eps):
    x0 = (xt - (1 - self.alpha_bar[t]).sqrt() * eps) / self.alpha_bar[t].sqrt()
    x0 = x0.clamp(-1, 1)
    # note that eps = (xt - sqrt(abar[t]) * x0) / sqrt(1 - abar[t])
    xt_1 = self.alpha_bar[max(0,t - self.ddim_mod)].sqrt() * x0 + (1 - self.alpha_bar[max(0,t - self.ddim_mod)]).sqrt() * eps
    return xt_1

  def ddpm_step(self, x, eps, t, var=None):
    var = self.beta[t] if var is None else var
    return (self.alpha[t]**-0.5) * (x - ((1 - self.alpha_bar[t])**0.5) * eps) + var * torch.randn_like(x)

  def generate(self, model, num_images=16, device="cuda"):
    with torch.no_grad():
      x = torch.randn((num_images, *self.shape), device=device) * self.std
      for t in range(self.T-1, -1, -self.ddim_mod):
        t_tensor = torch.full((num_images,),t, device=device)
        epsilons = model(x, t=t_tensor)
        x = self.ddim_step(x, t=t, eps=epsilons)
      return x