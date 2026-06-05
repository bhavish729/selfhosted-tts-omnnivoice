"""Generate a self-contained HTML report from burst_results.json.

The report has: a summary table across batch sizes, a TTFB-vs-batch chart, a
TTFB-distribution view, and per-batch expandable sections with a sortable
per-request table including an inline <audio> player for every request.

Audio is referenced relative to the report (audio/ subdir) so the whole
results/burst/ folder is portable — zip it and the report plays offline.
"""
from __future__ import annotations

import argparse
import json
import html
from pathlib import Path


def esc(s) -> str:
    return html.escape(str(s))


def build_html(data: dict) -> str:
    meta = data["meta"]
    batches = data["batches"]
    target = meta["ttfb_target_ms"]

    # summary rows
    sum_rows = ""
    for b in batches:
        cls = "good" if (b["under_target_pct"] or 0) >= 95 else ("warn" if (b["under_target_pct"] or 0) >= 50 else "bad")
        sum_rows += f"""<tr>
<td>{b['batch']}</td><td>{b['ok']}/{b['requests']}</td>
<td>{b['ttfb_p50']}</td><td>{b['ttfb_p95']}</td><td>{b['ttfb_p99']}</td><td>{b['ttfb_max']}</td>
<td class="{cls}">{b['under_target']} ({b['under_target_pct']}%)</td>
<td>{b['effective_rps']}</td><td>{b['wall_s']}</td><td>{b['errors']}</td></tr>"""

    # chart data
    labels = [b["batch"] for b in batches]
    p50 = [b["ttfb_p50"] for b in batches]
    p95 = [b["ttfb_p95"] for b in batches]
    p99 = [b["ttfb_p99"] for b in batches]
    mx = [b["ttfb_max"] for b in batches]
    under_pct = [b["under_target_pct"] for b in batches]

    # per-batch sections
    sections = ""
    for b in batches:
        rows = ""
        for r in sorted(b["requests_detail"], key=lambda x: x.get("ttfb_ms", 0)):
            ttfb = r.get("ttfb_ms")
            ok = r["status"] == 200
            tcls = "good" if (ttfb is not None and ttfb <= target) else "bad"
            audio_cell = ""
            if r.get("wav"):
                audio_cell = f'<audio controls preload="none" src="audio/{esc(r["wav"])}"></audio>'
            elif r.get("error"):
                audio_cell = f'<span class="bad">{esc(r["error"])}</span>'
            txt = esc(r["text"][:90]) + ("…" if len(r["text"]) > 90 else "")
            rows += f"""<tr>
<td>{r['idx']}</td><td><span class="lang">{esc(r['language_id'])}</span></td>
<td>{r['char_len']}</td>
<td class="{tcls}">{round(ttfb,1) if ttfb is not None else '-'}</td>
<td>{round(r.get('total_ms',0),1)}</td>
<td>{round(r.get('server_gen_ms',0),1)}</td>
<td>{r.get('audio_dur_s','-')}</td>
<td>{audio_cell}</td>
<td class="txt" title="{esc(r['text'])}">{txt}</td></tr>"""
        sections += f"""
<details>
<summary><b>Batch N={b['batch']}</b> — {b['ok']}/{b['requests']} ok · p50 {b['ttfb_p50']}ms · p95 {b['ttfb_p95']}ms · max {b['ttfb_max']}ms · {b['under_target']}/{b['ok']} under {int(target)}ms · {b['effective_rps']} RPS</summary>
<table class="detail">
<thead><tr><th>#</th><th>lang</th><th>chars</th><th>TTFB ms</th><th>total ms</th><th>gen ms</th><th>audio s</th><th>play</th><th>text</th></tr></thead>
<tbody>{rows}</tbody></table>
</details>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OmniVoice Burst Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}}
 h1{{margin:0 0 4px}} .sub{{color:#9aa4b2;margin-bottom:20px;font-size:14px}}
 table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
 th,td{{border:1px solid #2a2f3a;padding:6px 8px;text-align:right}}
 th{{background:#1a1f2b;position:sticky;top:0}}
 td.txt,th:last-child{{text-align:left}} td:nth-child(2),td:nth-child(1){{text-align:center}}
 .good{{color:#4ade80}} .warn{{color:#fbbf24}} .bad{{color:#f87171}}
 .lang{{background:#243; padding:1px 6px;border-radius:4px;font-size:11px}}
 details{{background:#161a22;border:1px solid #2a2f3a;border-radius:8px;margin:8px 0;padding:6px 12px}}
 summary{{cursor:pointer;padding:6px 0;font-size:14px}}
 audio{{height:28px;width:200px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
 .card{{background:#161a22;border:1px solid #2a2f3a;border-radius:10px;padding:14px 18px;min-width:150px}}
 .card .v{{font-size:24px;font-weight:700}} .card .l{{color:#9aa4b2;font-size:12px}}
 .chartbox{{background:#161a22;border:1px solid #2a2f3a;border-radius:10px;padding:16px;margin:14px 0;max-width:900px}}
</style></head><body>
<h1>OmniVoice Burst Benchmark</h1>
<div class="sub">Endpoint: {esc(meta['server_url'])} · target TTFB {int(target)} ms · pure-burst (all N fired simultaneously) · {meta['total_requests']} total requests</div>

<div class="cards">
 <div class="card"><div class="v">{esc(meta['batches'][0])}–{esc(meta['batches'][-1])}</div><div class="l">batch sizes</div></div>
 <div class="card"><div class="v">{sum(b['ok'] for b in batches)}</div><div class="l">successful requests</div></div>
 <div class="card"><div class="v">{sum(b['errors'] for b in batches)}</div><div class="l">errors</div></div>
 <div class="card"><div class="v">{max((b['ttfb_max'] or 0) for b in batches):.0f} ms</div><div class="l">worst TTFB</div></div>
</div>

<div class="chartbox"><canvas id="ttfb"></canvas></div>
<div class="chartbox"><canvas id="under"></canvas></div>

<h2>Summary by batch size</h2>
<table>
<thead><tr><th>N (burst)</th><th>ok</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>max ms</th>
<th>under {int(target)}ms</th><th>eff. RPS</th><th>wall s</th><th>err</th></tr></thead>
<tbody>{sum_rows}</tbody></table>

<h2>Per-request detail (with audio)</h2>
{sections}

<script>
const labels={json.dumps(labels)};
new Chart(document.getElementById('ttfb'),{{type:'line',
 data:{{labels:labels,datasets:[
  {{label:'p50',data:{json.dumps(p50)},borderColor:'#4ade80'}},
  {{label:'p95',data:{json.dumps(p95)},borderColor:'#fbbf24'}},
  {{label:'p99',data:{json.dumps(p99)},borderColor:'#fb923c'}},
  {{label:'max',data:{json.dumps(mx)},borderColor:'#f87171'}},
  {{label:'target {int(target)}ms',data:labels.map(_=>{int(target)}),borderColor:'#60a5fa',borderDash:[6,4],pointRadius:0}}
 ]}},
 options:{{plugins:{{title:{{display:true,text:'TTFB vs burst size',color:'#e6e6e6'}},legend:{{labels:{{color:'#e6e6e6'}}}}}},
  scales:{{x:{{title:{{display:true,text:'requests fired at once',color:'#9aa4b2'}},ticks:{{color:'#9aa4b2'}}}},
           y:{{title:{{display:true,text:'TTFB (ms)',color:'#9aa4b2'}},ticks:{{color:'#9aa4b2'}}}}}}}}}});
new Chart(document.getElementById('under'),{{type:'bar',
 data:{{labels:labels,datasets:[{{label:'% under target',data:{json.dumps(under_pct)},backgroundColor:'#4ade80'}}]}},
 options:{{plugins:{{title:{{display:true,text:'% of requests meeting TTFB target',color:'#e6e6e6'}},legend:{{labels:{{color:'#e6e6e6'}}}}}},
  scales:{{x:{{ticks:{{color:'#9aa4b2'}}}},y:{{max:100,ticks:{{color:'#9aa4b2'}}}}}}}}}});
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/burst/burst_results.json")
    ap.add_argument("--out", default="results/burst/report.html")
    args = ap.parse_args()
    data = json.load(open(args.results))
    Path(args.out).write_text(build_html(data), encoding="utf-8")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
