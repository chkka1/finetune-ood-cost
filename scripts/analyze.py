# -*- coding: utf-8 -*-
"""汇总实验、做统计并出图（按 case × 微调强度(tag) 分组）。

产出：
  results/analysis_summary.md   全部关键数字
  results/analysis.json         机器可读
  figures/fig2_dose.png         剂量-反应曲线（域内 vs 各域外；实线/虚线=不同微调强度）
  figures/fig3_gap.png          域外代价 vs 特征距离变化（核心图）
  figures/fig4_summary.png      各设置最终域内/域外增益对比
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300

CASE_ZH = {"pacs": "PACS（自然图像）", "organ": "OrganMNIST（腹部CT）",
           "digits": "Digits（手写/街景数字）"}
GRAYS = ["#000000", "#666666", "#AAAAAA", "#DDDDDD"]
MARKERS = ["o", "s", "^", "v"]
TAG_LABEL = {"": "lr=3e-4", "_strong": "lr=1e-3"}


def load():
    frames = []
    for p in sorted(RESULTS.glob("*.csv")):
        if "_smoke" in p.name:
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["file"] = p.stem
        frames.append(df)
    if not frames:
        raise SystemExit("results/ 下没有可用结果")
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["file"].str.extract(r"_s\d+(?P<tag>.*)$")["tag"]
    for col in ("case", "backbone", "size", "tag", "domain", "role"):
        df[col] = df[col].fillna("").astype(str)
    return df


def deltas(df):
    key = ["case", "backbone", "size", "tag", "seed", "domain"]
    base = df[df.epoch == 0].set_index(key)[["acc_ridge", "acc_nc", "mmd", "fr_dist"]]
    base.columns = [c + "_0" for c in base.columns]
    m = df.join(base, on=key)
    for c in ["acc_ridge", "acc_nc", "mmd", "fr_dist"]:
        m["d_" + c] = m[c] - m[c + "_0"]
    m["d_acc_pp"] = m["d_acc_ridge"] * 100.0
    m["d_mmd_rel"] = np.where(m["mmd_0"] > 0, m["d_mmd"] / m["mmd_0"], np.nan)
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = load()
    m = deltas(df)
    cases = [c for c in ["pacs", "organ", "digits"] if c in set(m["case"])]
    out = {"cases": cases, "per_case": {}}

    agg = (m.groupby(["case", "tag", "epoch", "domain", "role"])[
        ["acc_ridge", "d_acc_pp", "mmd", "d_mmd"]].agg(["mean", "std", "count"]).reset_index())
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.values]

    for c in cases:
        out["per_case"][c] = {}
        for tag in sorted(agg[agg["case"] == c]["tag"].unique()):
            sub = agg[(agg["case"] == c) & (agg["tag"] == tag)]
            rows = {}
            for dom in sorted(sub["domain"].unique()):
                s = sub[sub["domain"] == dom].sort_values("epoch")
                rows[dom] = {
                    "role": s["role"].iloc[0],
                    "epochs": s["epoch"].tolist(),
                    "acc_mean": s["acc_ridge_mean"].round(4).tolist(),
                    "d_acc_pp_mean": s["d_acc_pp_mean"].round(2).tolist(),
                    "d_acc_pp_std": s["d_acc_pp_std"].round(2).fillna(0).tolist(),
                    "mmd_mean": s["mmd_mean"].round(2).tolist(),
                    "d_mmd_mean": s["d_mmd_mean"].round(2).tolist(),
                    "n_seeds": int(s["acc_ridge_count"].max()),
                }
            out["per_case"][c][tag] = rows

    # ---------- 域外代价 vs 特征距离 ----------
    tgt = m[(m["role"] == "out") & (m["epoch"] > 0)].dropna(subset=["d_acc_pp", "d_mmd"])
    pr, pp = stats.pearsonr(tgt["d_mmd"], tgt["d_acc_pp"])
    sr, sp = stats.spearmanr(tgt["d_mmd"], tgt["d_acc_pp"])
    slope, intercept, rval, pval, stderr = stats.linregress(tgt["d_mmd"], tgt["d_acc_pp"])
    per_case = {}
    for c in cases:
        s = tgt[tgt["case"] == c]
        if len(s) > 2:
            r, p = stats.pearsonr(s["d_mmd"], s["d_acc_pp"])
            per_case[c] = {"n": int(len(s)), "pearson_r": round(float(r), 4), "p": float(p)}
    out["gap_vs_cost"] = {
        "n": int(len(tgt)), "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
        "spearman_r": round(float(sr), 4), "spearman_p": float(sp),
        "ols_slope": round(float(slope), 4), "ols_intercept": round(float(intercept), 3),
        "ols_r2": round(float(rval ** 2), 4), "ols_p": float(pval), "per_case": per_case,
    }

    # ---------- 各设置最终结果 ----------
    strongest = {}
    for c in cases:
        for tag in out["per_case"][c]:
            sub = m[(m["case"] == c) & (m["tag"] == tag)]
            if sub.empty:
                continue
            emax = int(sub["epoch"].max())
            s = sub[sub["epoch"] == emax]
            strongest["{}|{}".format(c, tag or "base")] = {
                "case": c, "tag": tag, "epoch": emax,
                "in_d_acc_pp": round(float(s[s["role"] == "in"]["d_acc_pp"].mean()), 2),
                "in_acc": round(float(s[s["role"] == "in"]["acc_ridge"].mean()), 4),
                "out_d_acc_pp": {dom: round(float(g["d_acc_pp"].mean()), 2)
                                 for dom, g in s[s["role"] == "out"].groupby("domain")},
                "out_mean_d_acc_pp": round(float(s[s["role"] == "out"]["d_acc_pp"].mean()), 2),
            }
    out["strongest"] = strongest

    (RESULTS / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    # ---------- 图 2 ----------
    fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.1), squeeze=False)
    for ax, c in zip(axes[0], cases):
        tags = sorted(out["per_case"][c])
        base_tag = "" if "" in tags else tags[0]
        for tag in tags:
            ls = "--" if (tag == base_tag and len(tags) > 1) else "-"
            for i, (dom, r) in enumerate(out["per_case"][c][tag].items()):
                ax.plot(r["epochs"], r["d_acc_pp_mean"], marker=MARKERS[i % 4], color=GRAYS[i % 4],
                        ls=ls, lw=2.0 if r["role"] == "in" else 1.4, ms=4.5,
                        label="{} {}{}".format(dom.split("(")[0],
                                               "域内" if r["role"] == "in" else "",
                                               "" if len(tags) == 1 else " " + TAG_LABEL.get(tag, tag)))
                ax.fill_between(r["epochs"],
                                np.array(r["d_acc_pp_mean"]) - np.array(r["d_acc_pp_std"]),
                                np.array(r["d_acc_pp_mean"]) + np.array(r["d_acc_pp_std"]),
                                color=GRAYS[i % 4], alpha=0.10, lw=0)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_title(CASE_ZH.get(c, c), fontsize=10)
        ax.set_xlabel("微调轮数", fontsize=9)
        ax.set_ylabel("相对未微调的精度变化 / 百分点", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
        ax.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_dose.png")
    plt.close()

    # ---------- 图 3 ----------
    plt.figure(figsize=(4.4, 3.4))
    for i, c in enumerate(cases):
        s = tgt[tgt["case"] == c]
        plt.scatter(s["d_mmd"], s["d_acc_pp"], s=17, marker=MARKERS[i % 4],
                    facecolors=GRAYS[i], edgecolors="black", linewidths=0.4,
                    label=CASE_ZH.get(c, c).split("（")[0])
    xs = np.linspace(tgt["d_mmd"].min(), tgt["d_mmd"].max(), 50)
    # 每个案例各自的拟合线（体现"案例内关系强、跨案例被结构差异削弱"）
    for i, c in enumerate(cases):
        s_c = tgt[tgt["case"] == c]
        if len(s_c) > 2:
            sl, ic, rv, pv, se = stats.linregress(s_c["d_mmd"], s_c["d_acc_pp"])
            xc = np.linspace(s_c["d_mmd"].min(), s_c["d_mmd"].max(), 30)
            plt.plot(xc, ic + sl * xc, color=GRAYS[i], lw=1.6, alpha=0.9)
    plt.plot(xs, intercept + slope * xs, color="black", lw=1.2, ls="--")
    plt.axhline(0, color="gray", lw=0.7); plt.axvline(0, color="gray", lw=0.7)
    plt.xlabel("域间特征距离变化（MMD，相对未微调）", fontsize=9)
    plt.ylabel("域外精度变化 / 百分点", fontsize=9)
    plt.title("虚线＝总体 r = {:.2f}（n = {}）；实线＝各案例拟合".format(pr, len(tgt)), fontsize=8.5)
    plt.legend(fontsize=7, frameon=False)
    plt.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_gap.png")
    plt.close()

    # ---------- 图 4 ----------
    labels = ["{}\n{}".format(CASE_ZH.get(v["case"], v["case"]).split("（")[0],
                              TAG_LABEL.get(v["tag"], v["tag"] or "base"))
              for v in strongest.values()]
    din = [v["in_d_acc_pp"] for v in strongest.values()]
    dout = [v["out_mean_d_acc_pp"] for v in strongest.values()]
    x = np.arange(len(labels)); w = 0.36
    plt.figure(figsize=(max(4.4, 1.35 * len(labels) + 1.6), 3.2))
    plt.bar(x - w / 2, din, w, color="white", edgecolor="black", lw=1.2, label="域内")
    plt.bar(x + w / 2, dout, w, color="#777777", edgecolor="black", lw=1.2, label="域外（均值）")
    for xi, v in zip(x - w / 2, din):
        plt.text(xi, v + (0.4 if v >= 0 else -1.8), "{:+.1f}".format(v), ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, dout):
        plt.text(xi, v + (0.4 if v >= 0 else -1.8), "{:+.1f}".format(v), ha="center", fontsize=8)
    plt.axhline(0, color="black", lw=0.8)
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("精度变化 / 百分点", fontsize=9)
    plt.legend(fontsize=8, frameon=False)
    plt.grid(axis="y", alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_summary.png")
    plt.close()

    # ---------- 文本 ----------
    L = ["# 汇总结果：域内/域外精度变化与特征距离", ""]
    for c in cases:
        L.append("## {}".format(CASE_ZH.get(c, c)))
        for tag, rows in out["per_case"][c].items():
            L.append("\n### 微调强度 {}".format(TAG_LABEL.get(tag, tag or "base")))
            for dom, r in rows.items():
                L.append("- **{}**（{}，n_seeds={}）".format(
                    dom, "域内" if r["role"] == "in" else "域外", r["n_seeds"]))
                L.append("  - 精度 {}".format(["%.3f" % v for v in r["acc_mean"]]))
                L.append("  - Δ(pp) {}".format(["%+.2f" % v for v in r["d_acc_pp_mean"]]))
                L.append("  - MMD {}".format(["%.1f" % v for v in r["mmd_mean"]]))
    gv = out["gap_vs_cost"]
    L.append("\n## 域外代价 vs 特征距离变化")
    L.append("- n={}，Pearson r={}（p={:.2e}），Spearman r={}（p={:.2e}）".format(
        gv["n"], gv["pearson_r"], gv["pearson_p"], gv["spearman_r"], gv["spearman_p"]))
    L.append("- OLS：Δacc = {:.3f} {}{:.3f}·ΔMMD，R²={}，p={:.2e}".format(
        gv["ols_intercept"], "+" if gv["ols_slope"] >= 0 else "", gv["ols_slope"],
        gv["ols_r2"], gv["ols_p"]))
    L.append("- 分案例：{}".format(gv["per_case"]))
    L.append("\n## 各设置最终结果（域内 vs 域外）")
    for v in strongest.values():
        L.append("- {} {}（epoch {}）：域内 {:+.2f} pp，域外均值 {:+.2f} pp；各域外 {}".format(
            CASE_ZH.get(v["case"], v["case"]), TAG_LABEL.get(v["tag"], v["tag"] or "base"),
            v["epoch"], v["in_d_acc_pp"], v["out_mean_d_acc_pp"], v["out_d_acc_pp"]))
    (RESULTS / "analysis_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
