import time, json, numpy as np, torch, soundfile as sf
import torch.nn as nn
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

TEXT = "आपके खाते में बारह हज़ार रुपये का भुगतान बकाया है।"
NS = 8
OUT = "/home/ubuntu/fp8bf.out"
def log(s):
    open(OUT,"a").write(s+"\n"); print(s, flush=True)

def bench(model, sr, tag, reps=12, save=None):
    cfg = OmniVoiceGenerationConfig(num_step=NS, guidance_scale=2.0)
    for _ in range(3):
        model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter()
        a=model.generate(text=TEXT, language="hi", instruct="female, indian accent", generation_config=cfg)[0]
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    arr=np.array(ts); rms=float(np.sqrt(np.mean(a**2)))
    log(f"{tag}: mean={arr.mean():.1f}ms p50={np.percentile(arr,50):.1f} p95={np.percentile(arr,95):.1f} rms={rms:.4f}")
    if save: sf.write(save, a, sr)
    return arr.mean(), rms

open(OUT,"w").write("")
log("loading bfloat16 model...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.bfloat16, load_asr=False)
sr = getattr(model, "sampling_rate", 24000)
res = {}
mbf, rbf = bench(model, sr, "BF16", save="/home/ubuntu/qbf_base.wav")
res["bf16_ms"]=mbf; res["bf16_rms"]=rbf

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
try:
    quantize_(model.llm, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=lambda m,f: isinstance(m, nn.Linear))
    quantize_(model.audio_heads, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=lambda m,f: isinstance(m, nn.Linear))
    torch.cuda.synchronize()
    m8, r8 = bench(model, sr, "FP8_FULL", save="/home/ubuntu/qbf_fp8.wav")
    res["fp8_ms"]=m8; res["fp8_rms"]=r8; res["fp8_speedup"]=round(mbf/m8,3)
except Exception as e:
    import traceback; log("FP8_FAIL "+repr(e)[:160]); log(traceback.format_exc()[-700:])
json.dump(res, open("/home/ubuntu/fp8bf_result.json","w"))
log("RESULT "+json.dumps(res)); log("DONE")
