import time, json, numpy as np, torch, soundfile as sf
import torch.nn as nn
from server import common

TEXT = "आपके खाते में बारह हज़ार रुपये का भुगतान बकाया है।"
NS = 8
OUT = "/home/ubuntu/fp8.out"
def log(s):
    open(OUT,"a").write(s+"\n")
    print(s, flush=True)

def bench(ctx, tag, reps=12, save=None):
    cfg = ctx.gen_cfg_cls(num_step=NS, guidance_scale=2.0)
    for _ in range(3):
        ctx.model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a = ctx.model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)[0]
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    arr = np.array(ts); rms = float(np.sqrt(np.mean(a**2)))
    log(f"{tag}: mean={arr.mean():.1f}ms p50={np.percentile(arr,50):.1f} p95={np.percentile(arr,95):.1f} rms={rms:.4f}")
    if save: sf.write(save, a, 24000)
    return arr.mean(), rms

open(OUT,"w").write("")
ctx = common.load_model(load_asr=False)
res = {}
res["llm_M"] = round(sum(p.numel() for p in ctx.model.llm.parameters())/1e6,1)
res["audio_heads_M"] = round(sum(p.numel() for p in ctx.model.audio_heads.parameters())/1e6,1)

m16, rms16 = bench(ctx, "FP16", save="/home/ubuntu/q_fp16.wav")
res["fp16_ms"]=m16; res["fp16_rms"]=rms16

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
def q(module):
    quantize_(module, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))
# LLM backbone (where the diffusion-step forwards run)
try:
    q(ctx.model.llm); torch.cuda.synchronize()
    m, r = bench(ctx, "FP8_LLM", save="/home/ubuntu/q_llm.wav")
    res["fp8_llm_ms"]=m; res["fp8_llm_rms"]=r; res["fp8_llm_speedup"]=round(m16/m,3)
except Exception as e:
    import traceback; log("LLM_FAIL "+repr(e)[:160]); log(traceback.format_exc()[-700:])
# + audio_heads
try:
    q(ctx.model.audio_heads); torch.cuda.synchronize()
    m, r = bench(ctx, "FP8_BOTH", save="/home/ubuntu/q_both.wav")
    res["fp8_both_ms"]=m; res["fp8_both_rms"]=r; res["fp8_both_speedup"]=round(m16/m,3)
except Exception as e:
    import traceback; log("HEAD_FAIL "+repr(e)[:160]); log(traceback.format_exc()[-700:])
json.dump(res, open("/home/ubuntu/fp8_result.json","w"))
log("RESULT "+json.dumps(res))
log("FP8_DONE")
