import torch
a = torch.randn(4,3,3).cuda()
u,s,v = torch.svd(a, some=False, compute_uv=True)
print(u)