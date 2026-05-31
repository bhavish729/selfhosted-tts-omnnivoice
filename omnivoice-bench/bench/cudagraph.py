import time, json, numpy as np, torch, soundfile as sf
from server import common

TEXT = "आपके खाते में बारह हज़ार रुपये का भुगतान बकाया है।"
NS = 8
OUT = "/home/ubuntu/cg.out"
def log(s):
    open(OUT,"a").write(s+"\n"); print(s, flush=True)

def bench(ctx, tag, reps=12, warm=5, save=None):
    cfg = ctx.gen_cfg_cls(num_step=NS, guidance_scale=2.0)
    for _ in range(warm):
        ctx.model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a = ctx.model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)[0]
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    arr = np.array(ts); rms = float(np.sqrt(np.mean(a**2)))
    log(f"{tag}: mean={arr.mean():.1f}ms p50={np.percentile(arr,50):.1f} p95={np.percentile(arr,95):.1f} min={arr.min():.1f} rms={rms:.4f}")
    if save: sf.write(save, a, 24000)
    return arr.mean(), rms

open(OUT,"w").write("")
torch.set_float32_matmul_precision("high")
ctx = common.load_model(load_asr=False)
res = {}
m_base, r_base = bench(ctx, "FP16_BASELINE", save="/home/ubuntu/cg_base.wav")
res["baseline_ms"] = m_base; res["baseline_rms"] = r_base

# Attempt 1: torch.compile reduce-overhead (CUDA graph trees) on the llm backbone
try:
    log("compiling llm with mode=reduce-overhead (CUDA graphs)...")
    ctx.model.llm = torch.compile(ctx.model.llm, mode="reduce-overhead", fullgraph=False)
    t0 = time.perf_counter()
    m_ro, r_ro = bench(ctx, "COMPILE_REDUCE_OVERHEAD", warm=8, save="/home/ubuntu/cg_ro.wav")
    log(f"(compile+warm took {time.perf_counter()-t0:.0f}s)")
    res["reduce_overhead_ms"]=m_ro; res["reduce_overhead_rms"]=r_ro; res["reduce_overhead_speedup"]=round(m_base/m_ro,3)
except Exception as e:
    import traceback; log("RO_FAIL "+repr(e)[:160]); log(traceback.format_exc()[-900:])

json.dump(res, open("/home/ubuntu/cg_result.json","w"))
log("RESULT "+json.dumps(res)); log("CG_DONE")
