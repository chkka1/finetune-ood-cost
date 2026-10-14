# -*- coding: utf-8 -*-
"""C4 稳健性检验：主干架构与输入分辨率的对照（用于正文 4.7 与表 7）。

三个配置统一取弱微调（lr = 3e-4）、16 轮、种子 0~2，在同一轮数上比较：
    ResNet-18 @128（主实验） / ResNet-18 @224 / ViT-B/16 @224
比较量：域内 Δacc、各域外域 Δacc 与其均值、以及标准化后的“距离-代价”斜率与 R²。

输出：results/robustness.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EPOCH = 16
SEEDS = (0, 1, 2)
OUT_DOMAINS = ("art_painting", "cartoon", "sketch")
CONFIGS = [
    ("ResNet-18@128", "pacs_resnet18_s128.csv", "main"),
    ("ResNet-18@224", "pacs_resnet18_s224_r224.csv", "res"),
    ("ViT-B/16@224", "pacs_vit_b_16_s224_v224.csv", "vit"),
]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    base = {}
    for r in rows:
        if int(r["epoch"]) == 0:
            base[(int(r["seed"]), r["domain"])] = {"acc": float(r["acc_ridge"]),
                                                   "fr": float(r["fr_dist"])}
    out = {}
    for r in rows:
        ep = int(r["epoch"])
        if ep == 0 or ep > EPOCH or int(r["seed"]) not in SEEDS:
            continue
        k = (int(r["seed"]), r["domain"])
        if k not in base:
            continue
        out[(k[0], k[1], ep)] = {"d_acc": 100.0 * (float(r["acc_ridge"]) - base[k]["acc"]),
                                 "d_fr": float(r["fr_dist"]) - base[k]["fr"],
                                 "final": ep == EPOCH}
    return out


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(y, x):
    A = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ b
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    se = np.sqrt(((y - yh) ** 2).sum() / (len(y) - 2) * np.linalg.inv(A.T @ A).diagonal())
    return b, se, float(r2)


def dom_mean(data, dom, key="d_acc"):
    """末轮、各种子平均。"""
    vals = [v[key] for (sd, d, ep), v in data.items() if dom in d and v.get("final")]
    return float(np.mean(vals)) if vals else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"epoch": EPOCH, "seeds": list(SEEDS), "configs": {}}
    print("=== 统一在弱微调第 %d 轮、种子 %s 上比较 ===" % (EPOCH, list(SEEDS)))
    print("  %-14s %8s %8s %8s %8s %9s %7s %5s" % (
        "配置", "域内", "Art", "Cartoon", "Sketch", "域外均值", "斜率", "R²"))
    for label, fname, tag in CONFIGS:
        data = load(fname)
        if not data:
            print("  %-14s （结果文件缺失：%s）" % (label, fname))
            continue
        ins = dom_mean(data, "photo")
        outs = {d: dom_mean(data, d) for d in OUT_DOMAINS}
        outs = {k: v for k, v in outs.items() if v is not None}
        ood_mean = float(np.mean(list(outs.values()))) if outs else None
        def fit(sel):
            pts = [(v["d_fr"], v["d_acc"]) for (sd, d, ep), v in data.items()
                   if "域内" not in d and "photo" not in d and sel(v)]
            if len(pts) <= 3:
                return None, None, None, len(pts)
            y = z([q[1] for q in pts]); x = z([q[0] for q in pts])
            b, se, r2 = ols(y, x)
            return float(b[1]), float(b[1] / se[1]), float(r2), len(pts)
        slope, t, r2, n_all = fit(lambda v: True)                 # 全部 ≤ EPOCH 轮
        ci = None
        if slope is not None and t:
            from scipy import stats as _st
            se_ = abs(slope / t)
            crit = float(_st.t.ppf(0.975, max(n_all - 2, 1)))
            ci = [round(slope - crit * se_, 2), round(slope + crit * se_, 2)]
        sfin, tfin, r2fin, n_fin = fit(lambda v: v["final"])      # 仅末轮
        cfg = {"in_domain": round(ins, 2) if ins is not None else None,
               "out_domains": {d: round(v, 2) for d, v in outs.items()},
               "out_mean": round(ood_mean, 2) if ood_mean is not None else None,
               "slope": round(slope, 3) if slope is not None else None,
               "slope_t": round(t, 1) if t is not None else None,
               "r2": round(r2, 3) if r2 is not None else None,
               "n_out_points": n_all,
               "slope_final": round(sfin, 3) if sfin is not None else None,
               "r2_final": round(r2fin, 3) if r2fin is not None else None,
               "ci95": ci,
               "n_final_points": n_fin,
               "n_seeds": len({sd for sd, _, _ in data})}
        out["configs"][label] = cfg
        print("  %-14s %8.2f %8.2f %8.2f %8.2f %9.2f %7.2f %5.2f" % (
            label, cfg["in_domain"], cfg["out_domains"].get("art_painting", float("nan")),
            cfg["out_domains"].get("cartoon", float("nan")), cfg["out_domains"].get("sketch", float("nan")),
            cfg["out_mean"], cfg["slope"] if cfg["slope"] is not None else float("nan"),
            cfg["r2"] if cfg["r2"] is not None else float("nan")))
    _a, _b = out["configs"].get("ResNet-18@128", {}), out["configs"].get("ResNet-18@224", {})
    if _a.get("slope") is not None and _b.get("slope") is not None:
        d = _b["slope"] - _a["slope"]
        se_d = float(np.hypot(abs(_a["slope"] / _a["slope_t"]), abs(_b["slope"] / _b["slope_t"])))
        from scipy import stats as _st
        se_d = float(np.hypot(abs(_a["slope"] / _a["slope_t"]), abs(_b["slope"] / _b["slope_t"])))
        out["slope_diff_128_vs_224"] = {"diff": round(d, 2),
                                        "ci95": [round(d - 1.96 * se_d, 2), round(d + 1.96 * se_d, 2)]}
        print("  两组 ResNet-18 斜率之差 %+.2f，95%%CI [%+.2f, %+.2f]" % (
            d, d - 1.96 * se_d, d + 1.96 * se_d))
    (RESULTS / "robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RESULTS / "robustness.json")


if __name__ == "__main__":
    main()
