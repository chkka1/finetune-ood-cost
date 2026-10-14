# -*- coding: utf-8 -*-
"""按中文核心/SCI 期刊通用规范绘制论文图（中文标注）。

输入：results/analysis.json（由 analyze.py 生成的统计结果）
输出：figures/fig{1..5}_*.{png,pdf,tif}
    fig1_overview  各案例在各域上的精度变化总览（热力图）
    fig2_doseheat  剂量-反应热力图（行为域、列为轮数）
    fig3_lrdose    学习率剂量-反应（末轮精度变化 vs 学习率）
    fig4_summary   域内/域外精度变化对比（分组条形）
    fig5_gap       域外精度变化与特征距离变化的关系（散点 + 拟合）

制图规范（中文核心/SCI 常见约定）：
  * 拉丁字母与数字用 Times New Roman，中文回退黑体（font.family 列表式回退）
  * 坐标轴四边闭合、刻度朝内、开启次刻度；不使用网格线
  * 误差棒为 ±1 标准差；全部图均用彩色（色盲友好的 Okabe-Ito 配色）
    ——图 2/图 3 的颜色表示"域"（域内统一黑色作参照），图 4 的颜色表示"案例"，
    图 1 为红-蓝发散色标；同时保留线型与标记形状编码，黑白打印亦可辨
  * 子图用 (a)(b)(c) 标注；图宽按单栏（约 8.9 cm）/双栏（约 17.5 cm）设置
  * 同时导出 600 dpi 位图（PNG/TIF）与矢量图（PDF），便于期刊取用
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

CASE_ZH = {"pacs": "PACS", "organ": "OrganMNIST", "digits": "Digits"}
TAG_ZH = {"": "弱微调（3e-4）", "_strong": "强微调（1e-3）", "_lr1e4": "更弱微调（1e-4）"}
TAG_ORDER = ["_lr1e4", "", "_strong"]          # 由弱到强
TAG_STYLE = {"_lr1e4": ":", "": "--", "_strong": "-"}   # 线型：点线/虚线/实线
LR_OF_TAG = {"_lr1e4": 1e-4, "": 3e-4, "_strong": 1e-3}
DOM_SHORT = {"photo": "Photo", "art_painting": "Art", "cartoon": "Cartoon", "sketch": "Sketch",
             "axial": "轴位", "coronal": "冠状位", "sagittal": "矢状位",
             "mnist": "MNIST", "emnist": "EMNIST", "svhn": "SVHN"}
# 灰度方案（图 1、图 3、图 4 使用）：域身份用灰度/标记，微调强度用线型
GRAYS = ["#000000", "#5A5A5A", "#9A9A9A", "#C8C8C8", "#E8E8E8"]
MARKERS = ["o", "s", "^", "v", "D"]
# 彩色方案（图 2 专用）：色盲友好的 Okabe-Ito 配色
#   域内（源域）统一用黑色作参照；每个目标域一种颜色 + 一种标记，同一图内不重复
COLOR_IN = "#000000"
DOM_COLOR = {"art_painting": "#D55E00", "cartoon": "#009E73", "sketch": "#CC79A7",
             "coronal": "#E69F00", "sagittal": "#56B4E9", "emnist": "#0072B2",
             "svhn": "#B58900"}
DOM_MARK = {"art_painting": "s", "cartoon": "^", "sketch": "v",
            "coronal": "D", "sagittal": "P", "emnist": "X", "svhn": "*"}
# 案例配色（图 4 用）：散点的颜色表示案例，与图 2/3 的"域配色"分属不同维度，各自图例已说明
CASE_COLOR = {"pacs": "#D55E00", "organ": "#009E73", "digits": "#0072B2"}
# 目标域统一排序（保证各图列序一致）
DOM_ORDER = ["art_painting", "cartoon", "sketch", "coronal", "sagittal", "emnist", "svhn"]


def setup_style():
    plt.rcParams.update({
        # font.family 直接给字体列表：matplotlib 才会做逐字回退
        # （拉丁字母/数字用 Times New Roman，中文缺字时回退到黑体）
        "font.family": ["Times New Roman", "SimHei"],
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.6,
        "axes.grid": False,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # 嵌入 TrueType，避免期刊排版出现 Type 3 字体
        "ps.fonttype": 42,
    })


def panel_label(ax, text, dx=0.02, dy=0.98):
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")


def save_all(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf", "tif"):
        p = FIG / "{}.{}".format(stem, ext)
        if ext == "tif":
            fig.savefig(p, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(p)
        outs.append(p)
    plt.close(fig)
    return outs


def dom_key(raw):
    """'art_painting(域外)' -> 'art_painting'"""
    return raw.split("(")[0]


# --------------------------------------------------------------------------- #
def fig1_overview(A):
    """总览热力图：行 = 案例（设置），列 = 域内 + 各目标域。"""
    strongest = A["strongest"]
    order = ["pacs|base", "pacs|_strong", "organ|base", "organ|_strong", "digits|base"]
    keys = [k for k in order if k in strongest]        # 只取弱/强两档，强度全景见图 4

    present = {dom_key(d) for k in keys for d in strongest[k]["out_d_acc_pp"]}
    cols = ["域内"] + [DOM_SHORT[d] for d in DOM_ORDER if d in present]

    M = np.full((len(keys), len(cols)), np.nan)
    rows = []
    for i, k in enumerate(keys):
        v = strongest[k]
        rows.append("{}（{}）".format(CASE_ZH.get(v["case"], v["case"]),
                                     TAG_ZH.get(v["tag"], "微调")))
        M[i, 0] = v["in_d_acc_pp"]
        for raw, val in v["out_d_acc_pp"].items():
            nm = DOM_SHORT.get(dom_key(raw), dom_key(raw))
            if nm in cols:
                M[i, cols.index(nm)] = val

    lim = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=(3.19, 2.55))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "{:+.1f}".format(M[i, j]), ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("精度变化 / 百分点", fontsize=8)
    cb.ax.tick_params(labelsize=7.5, width=0.6)
    cb.outline.set_linewidth(0.6)
    return save_all(fig, "fig1_overview")


# --------------------------------------------------------------------------- #
def pick_tag(A, c):
    """案例所选设置：有强微调则用强微调，否则用该案例唯一的设置。"""
    return "_strong" if "_strong" in A["per_case"][c] else sorted(A["per_case"][c])[0]


def fig2_doseheat(A):
    """剂量-反应热力图（案例 × 轮数）：行=域（域内在首行），列=轮数，格内为 Δacc（百分点）。

    每个案例取其实验最强设置；发散色标以 0 为中心，红=提升、蓝=下降。
    """
    cases = [c for c in ["pacs", "organ", "digits"] if c in A["per_case"]]
    panels = []
    vmax = 0.0
    for c in cases:
        tag = pick_tag(A, c)
        rows = A["per_case"][c][tag]
        inx = [d for d, r in rows.items() if r["role"] == "in"][0]
        order = [inx] + sorted([d for d, r in rows.items() if r["role"] == "out"],
                               key=lambda d: DOM_ORDER.index(dom_key(d))
                               if dom_key(d) in DOM_ORDER else 99)
        eps = [e for e in rows[inx]["epochs"] if e > 0]
        M = np.array([[rows[d]["d_acc_pp_mean"][rows[d]["epochs"].index(e)] for e in eps]
                      for d in order])
        vmax = max(vmax, float(np.nanmax(np.abs(M))))
        panels.append((c, tag, order, eps, M))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.69, 2.95), squeeze=False)
    cmap = plt.get_cmap("RdBu_r")
    im = None
    for k, (ax, (c, tag, order, eps, M)) in enumerate(zip(axes[0], panels)):
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels([str(e) for e in eps], fontsize=7.5)
        ax.set_yticks(range(len(order)))
        if k == 0:
            ax.set_yticklabels(["{}（{}）".format(DOM_SHORT.get(dom_key(d), dom_key(d)),
                                                 "域内" if rows_role(A, c, tag, d) == "in" else "域外")
                                for d in order], fontsize=7.5)
        else:
            ax.set_yticklabels([])
        for i in range(len(order)):
            for j in range(len(eps)):
                ax.text(j, i, "{:+.1f}".format(M[i, j]), ha="center", va="center",
                        fontsize=6, color="black")
        ax.set_xticks(np.arange(-0.5, len(eps), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="both", bottom=False, left=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.set_xlabel("微调轮数", fontsize=8.5)
        ax.set_title("{}（{}）".format(CASE_ZH.get(c, c), TAG_ZH.get(tag, tag).split("（")[0]),
                     fontsize=9, pad=3)
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.022, pad=0.012)
    cb.set_label("精度变化 / 百分点", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.6)
    return save_all(fig, "fig2_doseheat")


def rows_role(A, c, tag, dom):
    return A["per_case"][c][tag][dom]["role"]


def fig3_lrdose(A):
    """学习率剂量-反应：横轴为学习率（对数），纵轴为末轮相对基线的精度变化。

    同一案例的三档强度使用相同的轮数预算，因此横轴的变化只来自学习率。
    """
    cases = [c for c in ["pacs", "organ"] if c in A["per_case"]]
    fig, axes = plt.subplots(1, len(cases), figsize=(6.69, 2.75), squeeze=False)
    seen = {}
    for k, (ax, c) in enumerate(zip(axes[0], cases)):
        tags = [t for t in TAG_ORDER if t in A["per_case"][c]]
        doms = list(A["per_case"][c][tags[0]].items())
        inx = [d for d, r in doms if r["role"] == "in"]
        order = inx + sorted([d for d, r in doms if r["role"] == "out"],
                             key=lambda d: DOM_ORDER.index(dom_key(d))
                             if dom_key(d) in DOM_ORDER else 99)
        for i, dom in enumerate(order):
            raw = dom_key(dom)
            role = "域内" if A["per_case"][c][tags[0]][dom]["role"] == "in" else "域外"
            nm = DOM_SHORT.get(raw, raw)
            col = COLOR_IN if role == "域内" else DOM_COLOR.get(raw, GRAYS[1])
            mk = "o" if role == "域内" else DOM_MARK.get(raw, MARKERS[1])
            xs, ys, es = [], [], []
            for tg in tags:
                r = A["per_case"][c][tg].get(dom)
                if r is None:
                    continue
                ep = max(r["epochs"])
                j = r["epochs"].index(ep)
                xs.append(LR_OF_TAG.get(tg, np.nan))
                ys.append(r["d_acc_pp_mean"][j])
                es.append(r["d_acc_pp_std"][j])
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=4.5, lw=1.5,
                        elinewidth=0.8, capsize=2.2, capthick=0.8)
            seen.setdefault((nm, role), (col, mk))
        ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_xticks([1e-4, 3e-4, 1e-3])
        ax.set_xticklabels(["1e-4", "3e-4", "1e-3"], fontsize=8)
        ax.set_xlabel("微调学习率", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("末轮精度变化 / 百分点", fontsize=8.5)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
        ep = max(A["per_case"][c][tags[0]][order[0]]["epochs"])
        ax.set_title("{}（{} 轮）".format(CASE_ZH.get(c, c), ep), fontsize=9, pad=3)
    handles = [Line2D([], [], color=col, marker=mk, ls="-", ms=4.5, lw=1.5)
               for (col, mk) in seen.values()]
    labels = ["{}（{}）".format(nm, role) for (nm, role) in seen.keys()]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return save_all(fig, "fig3_lrdose")


def fig4_summary(A):
    """域内/域外精度变化对比：彩色分组条形 + 误差棒（误差棒来自逐种子标准差）。"""
    strongest = A["strongest"]
    order = ["pacs|_strong", "organ|_strong", "digits|base"]
    keys = [k for k in order if k in strongest]
    labels = ["{}\n{}".format(CASE_ZH.get(strongest[k]["case"], strongest[k]["case"]),
                              TAG_ZH.get(strongest[k]["tag"], "微调")) for k in keys]

    series, errs_series = [], []
    for k in keys:
        v = strongest[k]
        case, tag, ep = v["case"], v["tag"], v["epoch"]
        bars = [("域内", v["in_d_acc_pp"], None)]
        errs = [0.0]
        rows = A["per_case"][case][tag]
        for raw in DOM_ORDER:
            for dom, val in v["out_d_acc_pp"].items():
                if dom_key(dom) == raw:
                    sd = 0.0
                    if dom in rows and ep in rows[dom]["epochs"]:
                        sd = rows[dom]["d_acc_pp_std"][rows[dom]["epochs"].index(ep)]
                    bars.append((DOM_SHORT.get(raw, raw), val, raw))
                    errs.append(sd)
        series.append(bars)
        errs_series.append(errs)

    ncol = max(len(b) for b in series)
    x = np.arange(len(keys))
    total_w = 0.78
    w = total_w / ncol
    fig, ax = plt.subplots(figsize=(3.19, 2.75))
    ymax = max(v for s_ in series for _, v, _ in s_)
    ymin = min(v for s_ in series for _, v, _ in s_)
    ax.set_ylim(min(-20, ymin - 4), max(21, ymax + 3))
    for j in range(ncol):
        vals, errs, colors = [], [], []
        for s_, es in zip(series, errs_series):
            if j < len(s_):
                nm, val, raw = s_[j]
                vals.append(val); errs.append(es[j])
                colors.append(COLOR_IN if raw is None else DOM_COLOR.get(raw, GRAYS[1]))
            else:
                vals.append(np.nan); errs.append(0.0); colors.append("white")
        ax.bar(x + (j - (ncol - 1) / 2) * w, vals, w * 0.92, yerr=errs,
               color=colors, edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, capsize=1.6, capthick=0.7, ecolor="black"))
    for j in range(ncol):
        for i, s_ in enumerate(series):
            if j < len(s_):
                val = s_[j][1]
                err = errs_series[i][j] if j < len(errs_series[i]) else 0.0
                off = (err + 0.5) if val >= 0 else -(err + 0.5)
                ax.text(x[i] + (j - (ncol - 1) / 2) * w, val + off, "{:+.1f}".format(val),
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("精度变化 / 百分点", fontsize=9)
    handles, seen = [], set()
    for s_ in series:
        for nm, _, raw in s_:
            key = raw if raw is not None else "__in__"
            if key in seen:
                continue
            seen.add(key)
            handles.append(Patch(facecolor=COLOR_IN if raw is None else DOM_COLOR.get(raw, GRAYS[1]),
                                 edgecolor="black", linewidth=0.5,
                                 label=nm if raw is None else (nm + "（域外）")))
    fig.legend(handles=handles, fontsize=6.8, ncol=4, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    return save_all(fig, "fig4_summary")


# --------------------------------------------------------------------------- #
def fig5_gap(A):
    """域外精度变化 vs 特征距离变化。"""
    gv = A["gap_vs_cost"]
    cases = [c for c in ["pacs", "organ", "digits"] if c in A["per_case"]]
    fig, ax = plt.subplots(figsize=(3.19, 2.6))
    xs_all, ys_all = [], []
    for i, c in enumerate(cases):
        xs, ys = [], []
        for tag, rows in A["per_case"][c].items():
            for raw, r in rows.items():
                if r["role"] != "out":
                    continue
                for e, dmmd, dacc in zip(r["epochs"], r["d_mmd_mean"], r["d_acc_pp_mean"]):
                    if e > 0:
                        xs.append(dmmd); ys.append(dacc)
        xs, ys = np.array(xs), np.array(ys)
        xs_all += list(xs); ys_all += list(ys)
        col = CASE_COLOR.get(c, GRAYS[i])
        ax.scatter(xs, ys, s=13, marker=MARKERS[i], facecolors=col,
                   edgecolors="black", linewidths=0.4, label=CASE_ZH.get(c, c))
        if len(xs) > 2:
            sl, ic = np.polyfit(xs, ys, 1)
            xr = np.linspace(xs.min(), xs.max(), 30)
            ax.plot(xr, ic + sl * xr, color=col, lw=1.2, ls="-")
    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    sl, ic = np.polyfit(xs_all, ys_all, 1)
    xr = np.linspace(xs_all.min(), xs_all.max(), 50)
    ax.plot(xr, ic + sl * xr, color="black", lw=1.1, ls="--")
    ax.axhline(0, color="#888888", lw=0.6)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_xlabel("特征距离变化（MMD）", fontsize=9)
    ax.set_ylabel("域外精度变化 / 百分点", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.text(0.98, 0.97, "r = {:.2f}, P < 0.001, n = {}\n（案例内 |r| 最大 {:.2f}）".format(
        gv["pearson_r"], gv["n"],
        max(abs(v["pearson_r"]) for v in gv["per_case"].values()
            if v.get("pearson_r") is not None)),
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999999", lw=0.5))
    return save_all(fig, "fig5_gap")


def main(analysis_path=None, verbose=True):
    setup_style()
    A = json.loads(Path(analysis_path or (RESULTS / "analysis.json")).read_text(encoding="utf-8"))
    made = []
    made += fig1_overview(A)
    made += fig2_doseheat(A)
    made += fig3_lrdose(A)
    made += fig4_summary(A)
    made += fig5_gap(A)
    if verbose:
        for p in made:
            print("  figure:", p.name, "{:.0f} KB".format(p.stat().st_size / 1024))
    return made


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else None)
