import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from openpoints.models.backbone.pointvit_inv import InvFuncWrapper
from openpoints.models.backbone.pointvit_inv import Block_inv_F, Block_inv_G
from openpoints.models.backbone.pointvit_inv import RevBackProp, DropPath
from timm.models.layers import DropPath

use_AMP = False

cnt = 5
x1 = torch.randn([2, 128, 1024]).cuda(0) 
x2 = x1.clone().cuda(0)

drop_path = DropPath(0.2) # nn.Identity()#
mlist = nn.ModuleList([])
for i in range(cnt):
      fm = Block_inv_F(dim=384, num_heads=12, qkv_bias=False, drop=0., attn_drop=0.,
                        drop_path=drop_path, norm_args={'norm': 'ln'}).cuda(0)
      gm = Block_inv_G(dim=384, mlp_ratio=4., drop=0., drop_path=drop_path, 
                       norm_args={'norm': 'ln'},).cuda(0)
      block = InvFuncWrapper(fm, gm, split_dim=-1).cuda(0)
      mlist.append(block)


##########################################################################
# Option 1: Clear activations
##########################################################################
 
with torch.cuda.amp.autocast(enabled=use_AMP, dtype=torch.float16):
      conv1 = nn.Conv1d(128, 768, 1).cuda(0)
      xx1 = conv1(x1).permute(0, 2, 1)
      # xx = x
      y1 = RevBackProp.apply(xx1, mlist)
      z1 = torch.mean(y1)
      y1.retain_grad() 
      xx1.retain_grad() 
      z1.backward()

dxx1 = xx1.grad.clone()
mlist.zero_grad()


##########################################################################
# Option 2: Not clear activations
##########################################################################
mlist2 =  copy.deepcopy(mlist)
conv2 = copy.deepcopy(conv1)

with torch.cuda.amp.autocast(enabled=use_AMP, dtype=torch.float16):
      xx2 = conv2(x2).permute(0, 2, 1)
      # xx2 = x2
      xxx2 = xx2
      for i in range(cnt):
            xxx2 = mlist2[i](xxx2, lambd=1., alpha=0.)
            # xxx2 = mlist2[i](xxx2, lambd=0.1, alpha=1.)
            # xxx2 = mlist2[i](xxx2, lambd=0.5, alpha=0.5)
      y2 = xxx2
      z2 = torch.mean(y2)
      y2.retain_grad() 
      xx2.retain_grad() 
      device = 'cuda:0'
      print(f'Before backward: {torch.cuda.memory_allocated(device)}')
      z2.backward()
      print(f'After backward: {torch.cuda.memory_allocated(device)}')

      dxx2 = xx2.grad

print(torch.allclose(dxx1,dxx2, rtol=1e-05, atol=1e-10))
print(torch.allclose(dxx1,dxx2, rtol=1e-05, atol=1e-05))

print(torch.allclose(y1, y2, rtol=1e-05, atol=1e-20))
a = 1