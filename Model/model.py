import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttention(nn.Module):
  def __init__(self, in_c):
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=32, num_channels=in_c, eps=1e-6, affine=True)
    self.Q = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
    self.K = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
    self.V = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
    self.proj = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    b, c, h, w = x.shape
    R = self.norm(x)
    q, v, k = self.Q(R), self.V(R), self.K(R)
    q, v, k = q.reshape(b, c, h*w), v.reshape(b, c, h*w), k.reshape(b, c, h*w)
    q, v, k = q.permute(0, 2, 1), v, k
    R = torch.bmm(q, k) * (1.0 / math.sqrt(c))
    R = F.softmax(R, dim=2)
    R = torch.bmm(v, R)
    R = R.reshape(b, c, h, w)
    return self.proj(R) + x

class ResBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.reshape = False
    if in_c != out_c:
      self.reshape = True
      self.conv_reshape = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
    self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
    self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    if self.reshape:
      x = self.conv_reshape(x)
    res = x
    x = self.norm1(x)
    x = x * torch.sigmoid(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = x * torch.sigmoid(x)
    x = self.conv2(x)
    x = x + res
    return x

class Model(nn.Module):
  def __init__(self, T=1000, filters=[32, 64, 96, 128], depth=2, t_dim=512, LDM=False):
    super().__init__()
    self.t_dim = t_dim
    self.T = T
    self.conv_in = nn.Conv2d(4 + self.t_dim if LDM else 3 + self.t_dim, filters[0], kernel_size=1)
    self.down = nn.ModuleList([])
    for i in range(1,len(filters)):
      block = nn.Module()
      block.Blocks = nn.ModuleList([ResBlock(filters[i-1], filters[i])])
      for _ in range(1, depth):
        block.Blocks.append(ResBlock(filters[i], filters[i]))
      block.DownSample = nn.Conv2d(filters[i], filters[i], kernel_size=3, stride=2, padding=1)
      self.down.append(block)

    self.mid = nn.Sequential(ResBlock(filters[-1], filters[-1]),
                             SpatialAttention(filters[-1]),
                             ResBlock(filters[-1], filters[-1]))

    self.up = nn.ModuleList([])
    filters = filters[::-1]
    for i in range(1,len(filters)):
      block = nn.Module()
      block.Blocks = nn.ModuleList([ResBlock(filters[i-1]*2, filters[i])])
      for _ in range(1, depth):
        block.Blocks.append(ResBlock(filters[i], filters[i]))
      block.UpSample = nn.Upsample(scale_factor=2, mode="bilinear")
      self.up.append(block)
    self.conv_out = nn.Conv2d(filters[-1], 4 if LDM else 3, kernel_size=3, padding=1)

  def get_sinusoidal_emb(self, t):
    """ Recieves B 1 shaped t tensor with scalar timesteps, returns B D embeddings """
    freqs = torch.exp(-math.log(self.T) * torch.arange(start=0, end=self.t_dim // 2, dtype=torch.float32) / (self.t_dim // 2)).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

  def forward(self, x, t):
    t_emb = self.get_sinusoidal_emb(t)
    B, C, H, W = x.shape

    t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).expand(B, self.t_dim, H, W)
    x = torch.cat((x,t_emb), 1)
    x = self.conv_in(x)

    cache = []
    for block in self.down:
      for resblock in block.Blocks:
        x = resblock(x)
      cache.append(x.clone())
      x = block.DownSample(x)
    x = self.mid(x)
    for block in self.up:
      x = block.UpSample(x)
      x = torch.cat((x, cache.pop()), 1)
      for resblock in block.Blocks:
        x = resblock(x)

    return (self.conv_out(x))