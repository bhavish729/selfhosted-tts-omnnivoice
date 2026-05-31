import time, json, numpy as np, torch, soundfile as sf
from server import common

TEXTS = {
    "short": "धन्यवाद, आपका दिन शुभ रहे।",
    "med":   "आपके खाते में बारह हज़ार रुपये का भुगतान बकाया है।",
    "long":  "नमस्ते, मैं तारा बोल रही हूँ। आपके लोन की किस्त पंद्रह दिन से लंबित है, कृपया आज ही भुगतान कर दीजिए।",
}
NS = 8
OUT = "/home/ubuntu/cgf.out"
def log(s):
    open(OUT,"a").write(s+"\n"); print(s, flush=True)

def bench(ctx, key, text, tag, reps=10, warm=4):
    cfg = ctx.gen_cfg_cls(num_step=NS, guidance_scale=2.0)
    for _ in range(warm):
        ctx.model.generate(text=text, language="hi", instruct="female, indian accent", generation_config=cfg)
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter()
        a=ctx.model.generate(text=text, language="hi", instruct="female, indian accent", generation_config=cfg)[0]
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    arr=np.array(ts); dur=len(a)/24000
    log(f"{tag}[{key}]: mean={arr.mean():.1f}ms p95={np.percentile(arr,95):.1f} audio={dur:.2f}s rtf={arr.mean()/1000/dur:.4f}")
    return arr.mean()

open(OUT,"w").write("")
torch.set_float32_matmul_precision("high")
ctx = common.load_model(load_asr=False)
res={}
for k,t in TEXTS.items():
    res[f"base_{k}"]=bench(ctx,k,t,"BASELINE")
log("compiling llm reduce-overhead...")
ctx.model.llm = torch.compile(ctx.model.llm, mode="reduce-overhead", fullgraph=False)
for k,t in TEXTS.items():
    res[f"cg_{k}"]=bench(ctx,k,t,"CUDAGRAPH", warm=8)
for k in TEXTS:
    res[f"speedup_{k}"]=round(res[f"base_{k}"]/res[f"cg_{k}"],2)
json.dump(res, open("/home/ubuntu/cgf_result.json","w"))
log("RESULT "+json.dumps({k:round(v,1) if isinstance(v,float) else v for k,v in res.items()}))
log("CGF_DONE")
