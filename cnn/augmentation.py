from operations import Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from genotypes import Genotype
from model_search import Network as ModelNetwork
from kornia.augmentation import *



DA_2D_PRIMITIVES = [
    'none',
    'blur',
    'invert',
    'hflip',
    'vflip',
    'crop',
    'rotate',
]


# TODO: Implement augmentation operators
DA_2D_OPS = {
  'none' : lambda *args, **kwargs: Identity(),
  'blur' : lambda p: RandomBoxBlur(p=p), # p is proba
  'invert' : lambda p: RandomInvert(p=p),
  'hflip' : lambda p: RandomHorizontalFlip(p=p),
  'vflip' : lambda p: RandomVerticalFlip(p=p),
  'crop' : lambda p: RandomCrop(p=p),
  'rotate': lambda p: RandomRotation(p=p),
}

class DAMixedOp(nn.Module):

  def __init__(self, params):
    super(DAMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in DA_2D_PRIMITIVES:
      op = DA_2D_OPS[primitive](params)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class DACell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction):
    super(DACell, self).__init__()
    self.reduction = reduction

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      # For each step i, there are 2 + i incoming edges
      # Each incoming edge is an operation (a MixedOp instance).
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = DAMixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class DANetwork(nn.Module):

  def __init__(self, C, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(DANetwork, self).__init__()
    self._C = C
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      # nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = DACell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self._initialize_alphas()


  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    return s1


  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(DA_2D_PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != DA_2D_PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != DA_2D_PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((DA_2D_PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class UnifiedNetwork(nn.Module):

    def __init__(
        self,
        C, 
        layers, 
        criterion, 
        steps=4, 
        multiplier=4, 
        stem_multiplier=3,
        p=0.5
    ):
        super(UnifiedNetwork, self).__init__()

        self.augmentation = DANetwork(
          C, 
          layers, 
          criterion, 
          steps=steps, 
          multiplier=multiplier, 
          stem_multiplier=stem_multiplier,
          p=p
        )
        
        self.network = ModelNetwork(
          C, 
          layers, 
          criterion, 
          steps=steps, 
          multiplier=multiplier, 
          stem_multiplier=stem_multiplier
        )


    def forward(self, x):
        x = self.augmentation(x,)
        x = self.network(x)
        return x
    
    
    def arch_parameters(self):
        return self.augmentation.arch_parameters() + self.network.arch_parameters()
    
