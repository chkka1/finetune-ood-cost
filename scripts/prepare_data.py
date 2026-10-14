# -*- coding: utf-8 -*-
"""准备三个配对案例的公开数据，统一缓存为 npz。

案例（均为"同一批类别、不同来源/风格"的严格配对）：
  C1 PACS        : 7 类 × {photo, art_painting, cartoon, sketch}   自然图像
  C2 Organ       : 11 类 × {axial, coronal, sagittal}             医学影像（MedMNIST）
  C3 Digits      : 10 类 × {mnist, usps, emnist, svhn}            手写/街景数字

输出：data/cases/<case>.npz
   images uint8 [N,h,w,3]（灰度已复制为 3 通道）
   labels int64 [N]  （0..C-1，跨域统一）
   domains int64 [N] （域索引，顺序见 cases 定义）
   splits int64 [N]  （0=train 1=val 2=test）
另外为每个域输出一张抽样montage，便于人工确认图像朝向是否正确。
"""
from __future__ import annotations

import os
import sys
import ssl
import shutil
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CASES = DATA / "cases"
RAW = DATA / "raw"
for p in (DATA, CASES, RAW):
    p.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; research-data-collector/1.0)"}


# --------------------------------------------------------------------------- #
def save_case(name, images, labels, domains, splits, domain_names, class_names=None):
    path = CASES / (name + ".npz")
    np.savez_compressed(
        path,
        images=images.astype(np.uint8),
        labels=labels.astype(np.int64),
        domains=domains.astype(np.int64),
        splits=splits.astype(np.int64),
        domain_names=np.array(domain_names),
        class_names=np.array(class_names if class_names else []),
    )
    print("[{}] saved {}  images={} shape={}".format(name, path.name, len(images), images.shape))
    for d, dn in enumerate(domain_names):
        m = domains == d
        print("    {}: n={} train/val/test={}/{}/{}  类别数={}".format(
            dn, int(m.sum()),
            int((m & (splits == 0)).sum()), int((m & (splits == 1)).sum()),
            int((m & (splits == 2)).sum()), len(set(labels[m].tolist()))))
    montage(images, labels, domains, splits, name)
    return path


def montage(images, labels, domains, splits, name, per_row=10, rows=4):
    """每个域取前若干张拼图，便于人眼确认朝向/颜色是否正常。"""
    from PIL import Image
    out = []
    for d in sorted(set(domains.tolist())):
        idx = np.where((domains == d) & (splits == 0))[0][:per_row * rows]
        if len(idx) == 0:
            continue
        h, w = images.shape[1], images.shape[2]
        canvas = np.full((rows * h, per_row * w, 3), 255, np.uint8)
        for k, i in enumerate(idx):
            r, c = divmod(k, per_row)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
        out.append(canvas)
    if out:
        H = max(x.shape[0] for x in out)
        W = max(x.shape[1] for x in out)
        page = np.full((H * len(out), W, 3), 255, np.uint8)
        for i, x in enumerate(out):
            page[i * H:i * H + x.shape[0], :x.shape[1]] = x
        Image.fromarray(page).save(ROOT / "figures" / ("samples_" + name + ".png"))
        print("    montage -> figures/samples_{}.png".format(name))


# --------------------------------------------------------------------------- #
def prep_medmnist():
    """MedMNIST v2（Zenodo 10519652）：organ{a,c,s}mnist 同为 11 类、不同成像平面。"""
    srcs = [("organamnist", "axial"), ("organcmnist", "coronal"), ("organsmnist", "sagittal")]
    imgs, labs, doms, spls, names = [], [], [], [], []
    n_cls = None
    for d, (fn, dn) in enumerate(srcs):
        p = DATA / "medmnist" / (fn + ".npz")
        if not p.exists():
            p = RAW / "medmnist" / (fn + ".npz")
        if not p.exists():
            print("!! missing", p)
            return None
        z = np.load(p)
        for split, code in (("train", 0), ("val", 1), ("test", 2)):
            x = z[split + "_images"]
            y = z[split + "_labels"].astype(np.int64).ravel()
            if x.ndim == 3:
                x = np.repeat(x[..., None], 3, axis=3)
            if x.shape[-1] == 1:
                x = np.repeat(x, 3, axis=3)
            imgs.append(x.astype(np.uint8)); labs.append(y)
            doms.append(np.full(len(y), d, np.int64)); spls.append(np.full(len(y), code, np.int64))
        n_cls = len(set(labs[-3].tolist())) if n_cls is None else n_cls
        names.append(dn)
    return save_case("organ", np.concatenate(imgs), np.concatenate(labs),
                     np.concatenate(doms), np.concatenate(spls), names)


# --------------------------------------------------------------------------- #
def _torchvision_downloads(root):
    """用 torchvision 下载 MNIST / SVHN / EMNIST（自动重试），返回 torchvision 数据对象。"""
    from torchvision import datasets, transforms
    T = transforms.ToTensor()
    out = {}
    out["mnist"] = datasets.MNIST(root=str(root), train=True, download=True, transform=T)
    out["mnist_test"] = datasets.MNIST(root=str(root), train=False, download=True, transform=T)
    out["svhn"] = datasets.SVHN(root=str(root), split="train", download=True, transform=T)
    out["svhn_test"] = datasets.SVHN(root=str(root), split="test", download=True, transform=T)
    try:
        out["emnist"] = datasets.EMNIST(root=str(root), split="digits", train=True, download=True, transform=T)
        out["emnist_test"] = datasets.EMNIST(root=str(root), split="digits", train=False, download=True, transform=T)
    except Exception as exc:  # noqa: BLE001
        print("EMNIST 失败：", str(exc)[:120])
    return out


def _usps_manual(root):
    """USPS 镜像证书过期，用不校验证书的方式抓取，并放到 torchvision 期望的位置。"""
    dst = Path(root) / "USPS" / "raw"
    dst.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    base = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
    for remote, local in (("usps.bz2", "usps.bz2"), ("usps.t.bz2", "usps.t.bz2")):
        p = dst / local
        if p.exists() and p.stat().st_size > 10000:
            continue
        with urllib.request.urlopen(urllib.request.Request(base + remote, headers=UA),
                                    timeout=120, context=ctx) as r, open(p, "wb") as fh:
            shutil.copyfileobj(r, fh)
        print("  usps ->", p.name, p.stat().st_size)
    from torchvision import datasets, transforms
    T = transforms.ToTensor()
    return datasets.USPS(root=str(root), train=True, download=True, transform=T), \
           datasets.USPS(root=str(root), train=False, download=True, transform=T)


def prep_digits():
    """10 类数字四源：mnist / usps / emnist(digits) / svhn。"""
    from torchvision import transforms
    root = RAW / "digits"
    root.mkdir(parents=True, exist_ok=True)
    to_np = transforms.ToTensor()

    import torch as _t

    def arr(ds, limit=None):
        n = len(ds) if limit is None else min(limit, len(ds))
        X = np.zeros((n, 32, 32, 3), np.uint8)
        Y = np.zeros(n, np.int64)
        for i in range(n):
            x, y = ds[i]
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if x.shape[-1] != 32 or x.shape[-2] != 32:
                x = _t.nn.functional.interpolate(x.unsqueeze(0), size=(32, 32),
                                                 mode="bilinear", align_corners=False).squeeze(0)
            X[i] = (x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Y[i] = int(y)
        return X, Y

    parts = []
    dl = _torchvision_downloads(root)
    parts.append(("mnist", *arr(dl["mnist"], 20000), *arr(dl["mnist_test"])))
    # USPS：先手工下载再交给 torchvision 解析
    try:
        tr, te = _usps_manual(root)
        parts.append(("usps", *arr(tr), *arr(te)))
    except Exception as exc:  # noqa: BLE001
        print("USPS 失败：", str(exc)[:150])
    if "emnist" in dl:
        parts.append(("emnist", *arr(dl["emnist"], 20000), *arr(dl["emnist_test"])))
    parts.append(("svhn", *arr(dl["svhn"], 20000), *arr(dl["svhn_test"])))

    imgs, labs, doms, spls, names = [], [], [], [], []
    for d, (dn, xtr, ytr, xte, yte) in enumerate(parts):
        # 源域切成 train/val，其余域整体作为 test（跨源评测）
        n_val = len(xtr) // 10
        imgs += [xtr[n_val:], xtr[:n_val], xte]
        labs += [ytr[n_val:], ytr[:n_val], yte]
        spls += [np.zeros(len(ytr) - n_val, np.int64), np.ones(n_val, np.int64), np.full(len(yte), 2, np.int64)]
        doms += [np.full(len(xtr) - n_val, d, np.int64), np.full(n_val, d, np.int64), np.full(len(yte), d, np.int64)]
        names.append(dn)
    return save_case("digits", np.concatenate(imgs), np.concatenate(labs),
                     np.concatenate(doms), np.concatenate(spls), names)


# --------------------------------------------------------------------------- #
def prep_pacs():
    """PACS（HuggingFace flwrlabs/pacs，单 parquet；字段 image/domain/label）。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("!! 缺少 pyarrow，跳过 PACS（请用 venv 运行）")
        return None
    p = DATA / "pacs.parquet"
    if not p.exists():
        print("!! 缺少", p)
        return None
    t = pq.read_table(str(p), columns=None)
    dom = np.array(t.column("domain").to_pylist())
    lab = np.array(t.column("label").to_pylist(), np.int64)
    imgs_col = t.column("image").to_pylist()
    names = sorted(set(dom.tolist()))
    label_map = {v: i for i, v in enumerate(sorted(set(lab.tolist())))}
    X = np.zeros((len(lab), 128, 128, 3), np.uint8)
    from PIL import Image
    import io
    for i, item in enumerate(imgs_col):
        if isinstance(item, dict) and "bytes" in item:
            im = Image.open(io.BytesIO(item["bytes"])).convert("RGB").resize((128, 128), Image.BICUBIC)
        else:
            im = Image.open(io.BytesIO(item)).convert("RGB").resize((128, 128), Image.BICUBIC)
        X[i] = np.array(im)
    doms = np.array([names.index(d) for d in dom], np.int64)
    labs = np.array([label_map[v] for v in lab], np.int64)
    # 源域固定为 photo（PACS 的标准设定）：70/15/15 切 train/val/test；其余域整体作 test
    spls = np.full(len(labs), 2, np.int64)
    src = names.index("photo")
    idx = np.where(doms == src)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    n_val = int(round(len(idx) * 0.15))
    n_test = int(round(len(idx) * 0.15))
    spls[idx[n_val + n_test:]] = 0
    spls[idx[n_test:n_val + n_test]] = 1
    spls[idx[:n_test]] = 2
    return save_case("pacs", X, labs, doms, spls, names)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "medmnist"):
        prep_medmnist()
    if what in ("all", "digits"):
        prep_digits()
    if what in ("all", "pacs"):
        prep_pacs()
