import torch, torch.nn as nn
from server import common
ctx = common.load_model(load_asr=False)
m = ctx.model
print("TOP_CHILDREN:", [n for n,_ in m.named_children()])
print("TOTAL_PARAMS_M", round(sum(p.numel() for p in m.parameters())/1e6,1))
for n, c in m.named_children():
    pc = sum(p.numel() for p in c.parameters())
    if pc > 0:
        print(f"  child {n}: {pc/1e6:.1f}M")
lins = [(name, mod.in_features, mod.out_features) for name, mod in m.named_modules() if isinstance(mod, nn.Linear)]
print("NUM_LINEAR", len(lins))
lins.sort(key=lambda x: x[1]*x[2], reverse=True)
for name, i, o in lins[:6]:
    print(f"  LIN {name}: {i}x{o}")
