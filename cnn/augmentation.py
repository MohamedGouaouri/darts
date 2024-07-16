from operations import Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from genotypes import Genotype
from model_search import Network as ModelNetwork
from kornia.augmentation import *
from operations import FactorizedReduce, ReLUConvBN


DA_2D_PRIMITIVES = [
    'none',
    
    'blur_p0',
    'blur_p0.2',
    'blur_p0.4',
    'blur_p0.6',
    'blur_p0.8',
    
    'invert_p0',
    'invert_p0.2',
    'invert_p0.6',
    'invert_p0.8',
    
    'hflip_p0',
    'hflip_p0.2',
    'hflip_p0.4',
    'hflip_p0.6',
    'hflip_p0.8',
  
    'vflip_p0',
    'vflip_p0.2',
    'vflip_p0.4',
    'vflip_p0.6',
    'vflip_p0.8',
    
    # 'crop',
    'rotate_0',
    'rotate_45',
    'rotate_90',
    'rotate_135',
    'rotate_180',
    'rotate_225',
    'rotate_270',
    
]


# TODO: Implement augmentation operators
DA_2D_OPS = {
  'none' : lambda *args,: Identity(),
  
  'blur_p0' : lambda *args: RandomBoxBlur(p=0, keepdim=True),
  'blur_p0.2' : lambda *args: RandomBoxBlur(p=0.3, keepdim=True),
  'blur_p0.4' : lambda *args: RandomBoxBlur(p=0.4, keepdim=True),
  'blur_p0.6' : lambda *args: RandomBoxBlur(p=0.6, keepdim=True),
  'blur_p0.8' : lambda *args: RandomBoxBlur(p=0.8, keepdim=True),
  
  'invert_p0' : lambda *args: RandomInvert(p=0, keepdim=True),
  'invert_p0.2' : lambda *args: RandomInvert(p=0.2, keepdim=True),
  'invert_p0.4' : lambda *args: RandomInvert(p=0.4, keepdim=True),
  'invert_p0.6' : lambda *args: RandomInvert(p=0.6, keepdim=True),
  'invert_p0.8' : lambda *args: RandomInvert(p=0.8, keepdim=True),
  
  
  'hflip_p0' : lambda *args: RandomHorizontalFlip(p=0, keepdim=True,),
  'hflip_p0.2' : lambda *args: RandomHorizontalFlip(p=0.2, keepdim=True,),
  'hflip_p0.4' : lambda *args: RandomHorizontalFlip(p=0.4, keepdim=True,),
  'hflip_p0.6' : lambda *args: RandomHorizontalFlip(p=0.6, keepdim=True,),
  'hflip_p0.8' : lambda *args: RandomHorizontalFlip(p=0.8, keepdim=True,),
  
  
  'vflip_p0' : lambda *args: RandomVerticalFlip(p=0, keepdim=True,),
  'vflip_p0.2' : lambda *args: RandomVerticalFlip(p=0.2, keepdim=True,),
  'vflip_p0.4' : lambda *args: RandomVerticalFlip(p=0.4, keepdim=True,),
  'vflip_p0.6' : lambda *args: RandomVerticalFlip(p=0.6, keepdim=True,),
  'vflip_p0.8' : lambda *args: RandomVerticalFlip(p=0.8, keepdim=True,),
  
  
  # 'crop' : lambda p: RandomCrop(p=p),
  'rotate_0': lambda *args: RandomRotation(degrees=0, keepdim=True,),
  'rotate_45': lambda *args: RandomRotation(degrees=45.0, keepdim=True,),
  'rotate_90': lambda *args: RandomRotation(degrees=90.0, keepdim=True,),
  'rotate_135': lambda *args: RandomRotation(degrees=135.0, keepdim=True,),
  'rotate_180': lambda *args: RandomRotation(degrees=180.0, keepdim=True,),
  'rotate_225': lambda *args: RandomRotation(degrees=225.0, keepdim=True,),
  'rotate_270': lambda *args: RandomRotation(degrees=270, keepdim=True,),
  
}

class DAMixedOp(nn.Module):

  def __init__(self, p):
    super(DAMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in DA_2D_PRIMITIVES:
      op = DA_2D_OPS[primitive](p)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class DACell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, p):
    super(DACell, self).__init__()
    self.reduction = reduction

    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      # For each step i, there are 2 + i incoming edges
      # Each incoming edge is an operation (a MixedOp instance).
      for j in range(2+i):
        # stride = 2 if reduction and j < 2 else 1
        op = DAMixedOp(p)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      # for j, h in enumerate(states):
      #   print(f'Ops out: {self._ops[offset+j](h, weights[offset+j]).size()} h size: {h.size()}')
      # print('============================================================')
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class DANetwork(nn.Module):

  def __init__(self, C, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, p=0.5):
    super(DANetwork, self).__init__()
    self._C = C
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._p = p

    C_curr = stem_multiplier*C
    
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.conv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1)
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = DACell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self._p)
      reduction_prev = reduction
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
    return self.conv(s1)


  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(DA_2D_PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 
  
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
        num_classes,
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
          num_classes,
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
    
    def _loss(self, input, target):
      # r = torch.mean(torch.tensor([self.augmentation._loss(input, target), self.network._loss(input, target)]))
      
      return self.network._loss(input, target)