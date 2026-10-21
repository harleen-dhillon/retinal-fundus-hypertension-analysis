import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.draw import polygon
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

DRIVE, IDRID, JSIEC = Path("data/DRIVE"), Path("data/IDRiD"), Path("data/JSIEC")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
EPOCHS, POP, ITERS = 100, 20, 40           # lower EPOCHS (e.g. 5) for a quick test run
PRINTED_EQ45 = True                        # True = Eq. 45 exactly as printed in the paper; False = re-derived from Eq. 38 + 41
CV_REPEATS = 3                             # repeated 5-fold CV with different seeds (reports mean +- std)
np.random.seed(0); torch.manual_seed(0)


# ------------------------------------------------------------------ helpers
def load(p, mask=False):                                    # 256x256, as the Res-UNet input in Fig. 3
    im = Image.open(p).convert("L" if mask else "RGB").resize((256, 256), Image.NEAREST if mask else Image.BILINEAR)
    return (np.asarray(im) > 0).astype(np.uint8) if mask else np.asarray(im)   # > 0: some masks are stored as 0/1, not 0/255


def prep(rgb):                                              # Eq. 2: bilateral filter (parameters are my choice)
    return cv2.bilateralFilter(rgb, 9, 75, 75)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def metrics(y, p):                                          # Eq. 60-64 (+AUC)
    y = np.asarray(y); pr = np.asarray(p) > .5
    tp, tn, fp, fn = (pr & (y == 1)).sum(), (~pr & (y == 0)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum()
    d = lambda a, b: round(float(a / b) if b else 0., 3)
    return dict(acc=d(tp + tn, len(y)), sens=d(tp, tp + fn), spec=d(tn, tn + fp), ppv=d(tp, tp + fp), npv=d(tn, tn + fn),
                auc=round(float(roc_auc_score(y, p)), 3))


# ------------------------------------------------------------------ the paper's two optimisers (minimise fitness)
def ogo(fit, lb, ub, pop=20, iters=40, b=1.0, init=None, seed=0):
    """Osprey-Gannet Optimization, Eq. 13-31. Assumed (not in paper): b=1, V1~U(0,1), keep a move only if it improves."""
    rng = np.random.default_rng(seed); lb, ub = np.asarray(lb, float), np.asarray(ub, float); n = lb.size
    Z = lb + rng.random((pop, n)) * (ub - lb)                                 # Eq. 13
    if init is not None: Z[0] = init
    F = np.array([fit(z) for z in Z]); hist = [F.min()]
    for r in range(1, iters + 1):
        for m in range(pop):
            fish = np.vstack([Z[F < F[m]], Z[F.argmin()][None]])              # Eq. 14: better solutions + best
            G = fish[rng.integers(len(fish))]
            t, J = rng.random(n), rng.integers(1, 3, n)
            C, V1, Zt = (2 * rng.random() - 1) * b, rng.random(), Z[rng.integers(pop)]
            den = C + t * J; den = np.where(np.abs(den) < 1e-6, 1e-6, den)
            Z1 = np.clip(((C * Zt - V1) * (1 - t * J) + t * G * (1 + C)) / den, lb, ub)   # Eq. 28 (OOA fused with GOA), Eq. 29
            f1 = fit(Z1)
            if f1 < F[m]: Z[m], F[m] = Z1, f1
            Z2 = np.clip(Z[m] + (lb + rng.random(n) * (ub - lb)) / r, lb, ub)  # Eq. 30-31 exploitation
            f2 = fit(Z2)
            if f2 < F[m]: Z[m], F[m] = Z2, f2
        hist.append(F.min())
    return Z[F.argmin()].copy(), hist


def jaya_ccboa(fit, lb, ub, pop=20, iters=40, init=None, seed=0):
    """Jaya Chronological Chef-Based Optimization, Eq. 34-45 (Eq. 45 as printed unless PRINTED_EQ45 is False)."""
    rng = np.random.default_rng(seed); lb, ub = np.asarray(lb, float), np.asarray(ub, float); n = lb.size
    Q = lb + rng.random((pop, n)) * (ub - lb)                                 # Eq. 35
    if init is not None: Q[0] = init
    F = np.array([fit(q) for q in Q]); Qp = Q.copy(); k = max(2, pop // 10); hist = [F.min()]
    for _ in range(iters):
        o = np.argsort(F); Q, F, Qp = Q[o], F[o], Qp[o]                       # Eq. 36: instructors first
        best, worst = Q[0].copy(), Q[-1].copy()
        for l in range(pop):
            if l < k:                                                         # chef instructors, Eq. 37
                new = Q[l] + rng.random(n) * (best - rng.integers(1, 3, n) * Q[l])
            else:                                                             # cooking students, Eq. 38-45
                r1, r2, rr, T = rng.random(n), rng.random(n), rng.uniform(-1, 1, n), rng.uniform(1, 2, n)
                TK = Q[rng.integers(k)]; A, B = 1 - r1 + r2, 1 - rr * T; den = 2 * A - B
                den = np.where(np.abs(den) < 1e-6, 1e-6, den)
                new = ((r2 * worst - r1 * best) * B + A * Qp[l] * B ** 2 + (1 if PRINTED_EQ45 else A) * rr * T * TK * (3 - rr * T)) / den
            new = np.clip(new, lb, ub); fn = fit(new); Qp[l] = Q[l]
            if fn < F[l]: Q[l], F[l] = new, fn
        hist.append(F.min())
    return Q[F.argmin()].copy(), hist


# ------------------------------------------------------------------ optic disc: snake (Eq. 3-10)
DEFAULT = (0.01, 0.1, 0.0, 1.0)             # alpha (elasticity), beta (stiffness), w_line, w_edge


def centre(rgb):                            # start point = brightest smoothed blob inside the retina (my assumption)
    g = rgb[..., :2].mean(2) / 255.; s = gaussian(g, 6) * ndi.binary_erosion(g > .08, iterations=10)
    return tuple(int(v) for v in np.unravel_index(s.argmax(), s.shape))


def snake(rgb, p, c):
    img = gaussian(rgb[..., :2].mean(2) / 255., 2)                            # Eq. 10: Gaussian smoothing
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    init = np.stack([c[0] + 23 * np.sin(th), c[1] + 23 * np.cos(th)], 1)
    s = active_contour(img, init, alpha=p[0], beta=p[1], w_line=p[2], w_edge=p[3], gamma=0.01, max_num_iter=150)
    m = np.zeros(img.shape, np.uint8); m[polygon(s[:, 0], s[:, 1], img.shape)] = 1
    return m


def stage_od():
    data = {"train": [], "test": []}

    image_root = IDRID / "images"
    mask_root = IDRID / "masks"
    archive_root = Path("archive")

    for gt in sorted(mask_root.glob("*_OD.tif")):
        image_name = gt.name.replace("_OD.tif", ".jpg")
        img = image_root / image_name

        if not img.exists():
            print(f"[OD] Skipping missing image: {image_name}")
            continue

        # Recover the official split from the original archive.
        original_masks = list(archive_root.rglob(gt.name))
        if not original_masks:
            print(f"[OD] Cannot find original mask: {gt.name}")
            continue

        original_path = str(original_masks[0]).lower()
        split = "train" if "training set" in original_path else "test"

        rgb = prep(load(img))
        mask = load(gt, True)

        if mask.shape != rgb.shape[:2]:
            print(f"[OD] Skipping size mismatch: {image_name}")
            continue

        data[split].append((rgb, mask, centre(rgb)))

    train = data["train"]
    test = data["test"]

    print(f"[OD] {len(train)} train / {len(test)} test images")

    assert train and test, (
        "No training or testing image-mask pairs found. "
        "Check the original archive paths."
    )
    assert all(mask.sum() > 0 for _, mask, _ in train + test), (
        "At least one optic-disc mask is empty."
    )

    win = lambda m, c: m[
        max(c[0] - 32, 0):c[0] + 32,
        max(c[1] - 32, 0):c[1] + 32
    ].astype(float)

    fit = lambda p: float(np.mean([
        ((win(snake(rgb, p, c), c) - win(mask, c)) ** 2).mean()
        for rgb, mask, c in train[:20]
    ]))

    dice = lambda p: float(np.mean([
        2 * (snake(rgb, p, c) & mask).sum()
        / (snake(rgb, p, c).sum() + mask.sum() + 1e-9)
        for rgb, mask, c in test
    ]))

    best, hist = ogo(
        fit,
        [0, 0, -0.5, 0],
        [0.5, 5, 1, 3],
        pop=8,
        iters=8,
        init=np.array(DEFAULT)
    )

    print(
        f"[OD] Test Dice: default={dice(DEFAULT):.3f}, "
        f"OGO-tuned={dice(best):.3f}; "
        f"weights={np.round(best, 3)}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "od_params.json", "w") as f:
        json.dump({"params": best.tolist()}, f)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    for a_, (rgb, mask, c) in zip(ax.ravel(), test[:6]):
        predicted = snake(rgb, best, c)
        a_.imshow(rgb)
        a_.contour(mask, [.5], colors="lime")
        a_.contour(predicted, [.5], colors="yellow")
        a_.axis("off")
        score = 2 * (predicted & mask).sum() / (
            predicted.sum() + mask.sum() + 1e-9
        )
        a_.set_title(f"Dice {score:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "od.png", dpi=110)
    plt.close(fig)


class Blk(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(o, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
        )
        self.skip = nn.Conv2d(i, o, 1)

    def forward(self, x):
        return torch.relu(self.f(x) + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, f=32):
        super().__init__()
        self.e1 = Blk(3, f)
        self.e2 = Blk(f, 2 * f)
        self.b = Blk(2 * f, 4 * f)
        self.d2 = Blk(6 * f, 2 * f)
        self.d1 = Blk(3 * f, f)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)
        self.head = nn.Conv2d(f, 2, 1)

    def feats(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.up(d2), e1], dim=1))
        return d1

    def forward(self, x):
        return self.head(self.feats(x))



def stage_bvs():
    X, Y, M = [], [], []                    # only the 20 labelled DRIVE training images have vessel masks
    for f in sorted((DRIVE / "training" / "images").glob("*.tif")):
        n = f.stem.split("_")[0]
        X.append(prep(load(f))); Y.append(load(next((DRIVE / "training" / "1st_manual").glob(n + "_*")), True))
        M.append(load(next((DRIVE / "training" / "mask").glob(n + "_*")), True))
    X = torch.tensor(np.stack(X)).permute(0, 3, 1, 2).float() / 255; Y = torch.tensor(np.stack(Y)).long(); M = np.stack(M).astype(bool)
    idx = np.random.permutation(len(X)); va, tr = idx[:4], idx[4:]
    print(f"[BVS] {len(tr)} train / {len(va)} validation images")
    net = ResUNet(); opt = torch.optim.Adam(net.parameters(), 1e-3); CW = [1., 4.]; ce = nn.CrossEntropyLoss(weight=torch.tensor(CW))
    for ep in range(EPOCHS):                # stage A: Adam pre-training
        net.train()
        for b in np.array_split(np.random.permutation(tr), 4):
            loss = ce(net(X[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0: print(f"  epoch {ep} loss {loss.item():.3f}")

    def evaluate():
        net.eval()
        with torch.no_grad(): P = torch.softmax(net(X[va]), 1)[:, 1].numpy()
        g, f = Y[va].numpy(), M[va]; m = metrics(g[f], P[f])
        m["dice"] = round(float(2 * ((P > .5) & (g == 1) & f).sum() / (((P > .5) & f).sum() + ((g == 1) & f).sum())), 3)
        return m, P
    print("[BVS] validation, Adam only        :", evaluate()[0])

    net.eval()                              # stage B: Jaya-CCBOA tunes the 1x1 output layer on cached features
    with torch.no_grad(): F = net.feats(X[tr]).numpy()
    rng = np.random.default_rng(0); Fp, Tp = [], []
    for i, k in enumerate(tr):
        ys, xs = np.nonzero(M[k]); s = rng.choice(len(ys), 20000, replace=False)
        Fp.append(F[i][:, ys[s], xs[s]].T); Tp.append(Y[k].numpy()[ys[s], xs[s]])
    Fp, Tp = np.concatenate(Fp).astype(np.float32), np.concatenate(Tp); c = Fp.shape[1]
    th0 = np.r_[net.head.weight.detach().numpy().ravel(), net.head.bias.detach().numpy()]
    T1, w = np.eye(2)[Tp], np.where(Tp == 1, CW[1], CW[0])

    def fit(th):                            # Eq. 33: mean squared error between target and Res-UNet output (class-weighted)
        th = th.astype(np.float32); s = softmax(Fp @ th[:-2].reshape(2, c).T + th[-2:])
        return float((w[:, None] * (T1 - s) ** 2).mean())
    best, hist = jaya_ccboa(fit, th0 - .5, th0 + .5, POP, ITERS, init=th0)
    with torch.no_grad():
        net.head.weight.copy_(torch.tensor(best[:-2]).reshape(2, c, 1, 1).float()); net.head.bias.copy_(torch.tensor(best[-2:]).float())
    m, P = evaluate()
    print("[BVS] validation, Adam + Jaya-CCBOA:", m)
    torch.save(net.state_dict(), OUT / "resunet.pt")
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for j in range(4):
        for r, im in enumerate([X[va[j]].permute(1, 2, 0).numpy(), Y[va[j]].numpy(), P[j] * M[va[j]]]):
            ax[r, j].imshow(im, cmap="gray" if r else None); ax[r, j].axis("off")
    ax[0, 0].set_title("input"); ax[1, 0].set_title("ground truth"); ax[2, 0].set_title("Res-UNet vessel probability")
    plt.savefig(OUT / "bvs.png", dpi=110)


# ------------------------------------------------------------------ features (Eq. 46-57) and Tanimoto (Eq. 56)
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
sh = lambda g, dy, dx: np.roll(g, (-dy, -dx), (0, 1))        # value of neighbour (dy, dx)


def stats6(v):                              # Eq. 46-51: mean, variance, std, skewness, kurtosis, energy
    v = np.asarray(v, float).ravel()
    if v.size == 0: return [0.] * 6
    mu = v.mean(); sd = v.std()
    return [mu, v.var(), sd, ((v - mu) ** 3).mean() / (sd ** 3 + 1e-12), ((v - mu) ** 4).mean() / (sd ** 4 + 1e-12), np.sqrt((v ** 2).sum())]


def ldtp(g):                                # c1 (Eq. 53, simplified): 16*direction + 4*P(L1) + P(L2), ternary-coded differences
    nb = np.stack([sh(g, *d) - g for d in RING]); d = np.abs(nb).argmax(0)
    q = lambda x: np.where(np.abs(x) <= 5, 0, np.where(x > 0, 1, 2))
    return 16 * d + 4 * q(np.take_along_axis(nb, d[None], 0)[0]) + q(np.take_along_axis(nb, ((d + 4) % 8)[None], 0)[0])


def phog(g):                                # c2: gradient-orientation histograms over a 1 + 4 + 16 cell pyramid -> 21 x 8 matrix
    gy, gx = np.gradient(g); mag = np.hypot(gx, gy); b = np.minimum((np.arctan2(gy, gx) % np.pi / np.pi * 8).astype(int), 7)
    rows = []
    for n in (1, 2, 4):
        for i in range(n):
            for j in range(n):
                s = (slice(i * 256 // n, (i + 1) * 256 // n), slice(j * 256 // n, (j + 1) * 256 // n))
                h = np.bincount(b[s].ravel(), weights=mag[s].ravel(), minlength=8); rows.append(h / (h.sum() + 1e-12))
    return np.array(rows)


def lvp(g):                                 # c3 (simplified): 8-bit code from comparing the derivative at the centre with its neighbours
    v = g - sh(g, 0, 1)
    return sum(((sh(v, *d) - v >= 0).astype(int) << k) for k, d in enumerate(RING))


def flbp(g):                                # c4: fuzzy LBP; weight of a pattern = product of memberships (Eq. 54)
    code, W = 0, 1.
    for k, d in enumerate(RING):
        mu = np.clip(.5 + (sh(g, *d) - g) / 10., 0, 1); code = code + ((mu >= .5).astype(int) << k); W = W * np.maximum(mu, 1 - mu)
    return code * W


def glcm_feats(rgb):                        # F2: GLCM homogeneity (Eq. 55) + entropy (Eq. 56) of c1..c4 = 8 numbers
    g, out = rgb2gray(rgb) * 255., []
    for m in (ldtp(g), phog(g), lvp(g), flbp(g)):
        m = ((m - m.min()) / (np.ptp(m) + 1e-12) * 31).astype(np.uint8)
        P = graycomatrix(m, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 32, symmetric=True, normed=True)
        out += [float(graycoprops(P, "homogeneity").mean()), float((-(P[:, :, 0] * np.log2(P[:, :, 0] + 1e-12)).sum(axis=(0, 1))).mean())]
    return out


def tanimoto(a, b):                         # standard form; the paper prints '+sum(ab)' in the denominator (identical vectors -> 1/3)
    ab = (a * b).sum(); return ab / ((a * a).sum() + (b * b).sum() - ab + 1e-12)


def select(X, thr=.9):                      # keep a feature unless it is >90 % Tanimoto-similar to one already kept (my rule)
    keep = []
    for j in range(X.shape[1]):
        if X[:, j].std() > 1e-9 and all(tanimoto(X[:, j], X[:, k]) < thr for k in keep): keep.append(j)
    return keep


# ------------------------------------------------------------------ classifier: CNN (Fig. 5) + OGO on its output layer
class FeatCNN(nn.Module):                   # Fig. 5: [Conv -> ReLU -> Dropout -> BN] blocks with 4 max-pools, then Dense256 -> Dense128 -> Dense2
    def __init__(s, L):                     # L = number of selected features (the figure's 200 is not reachable from the paper's own feature list)
        super().__init__()
        conv = lambda i, o, pool: [nn.Conv1d(i, o, 3, padding=1), nn.ReLU(), nn.Dropout(.2), nn.BatchNorm1d(o)] + ([nn.MaxPool1d(2, ceil_mode=True)] if pool else [])
        dense = lambda i, o: [nn.Linear(i, o), nn.ReLU(), nn.Dropout(.2), nn.BatchNorm1d(o)]
        for _ in range(4): L = (L + 1) // 2                      # length after four ceil-mode max-pools
        s.body = nn.Sequential(*conv(1, 32, 0), *conv(32, 64, 1), *conv(64, 128, 1), *conv(128, 128, 1), *conv(128, 128, 1),
                               nn.Flatten(), nn.Dropout(.2), *dense(128 * L, 256), *dense(256, 128))   # the 3 middle conv blocks are inferred from the printed shapes
        s.out = nn.Linear(128, 2)           # 'Dense_5' in Fig. 5: the layer OGO tunes (258 numbers)

    def forward(s, x): return s.out(s.body(x))


def fit_model(Xtr, ytr, seed=0):                    # scale -> Tanimoto selection -> Adam (stand-in for LeNet pre-training) -> OGO on output layer
    lo, hi = Xtr.min(0), Xtr.max(0); sc = lambda Z: np.clip((Z - lo) / (hi - lo + 1e-9), 0, 1)
    torch.manual_seed(seed); keep = select(sc(Xtr)); Xs = torch.tensor(sc(Xtr)[:, keep], dtype=torch.float32)[:, None, :]; yt = torch.tensor(ytr)
    cw = len(ytr) / (2 * np.bincount(ytr)); net = FeatCNN(len(keep)); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    for _ in range(150):
        net.train()
        for b in torch.randperm(len(Xs)).split(16):
            if len(b) > 1: loss = ce(net(Xs[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): H = net.body(Xs).numpy()
    th0 = np.r_[net.out.weight.detach().numpy().ravel(), net.out.bias.detach().numpy()]
    T1 = np.eye(2)[ytr]
    fit = lambda th: float(((T1 - softmax(H @ th[:-2].reshape(2, -1).T + th[-2:])) ** 2).mean())   # Eq. 59: plain MSE, as in the paper
    th, hist = ogo(fit, th0 - .5, th0 + .5, POP, ITERS, init=th0, seed=seed)
    def predict(Z, th):
        with torch.no_grad(): Hz = net.body(torch.tensor(sc(Z)[:, keep], dtype=torch.float32)[:, None, :]).numpy()
        return softmax(Hz @ th[:-2].reshape(2, -1).T + th[-2:])[:, 1]
    return predict, th0, th, hist, net


def stage_cls():
    net = ResUNet(); net.load_state_dict(torch.load(OUT / "resunet.pt", map_location="cpu")); net.eval()
    par = json.load(open(OUT / "od_params.json"))["params"] if (OUT / "od_params.json").exists() else DEFAULT
    items = []
    for d in sorted(p for p in JSIEC.rglob("*") if p.is_dir()):
        lab = 0 if "normal" in d.name.lower() else 1 if "hypertensive" in d.name.lower() else None
        if lab is not None: items += [(f, lab) for f in sorted(d.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}]
    y = np.array([l for _, l in items]); print(f"[CLS] {int((y == 0).sum())} normal, {int((y == 1).sum())} hypertensive images")
    X = []
    for f, _ in items:                      # per image: 6 stats of OD region + 6 of vessel map + 8 GLCM features = 20 features
        raw = load(f); r = prep(raw); od = snake(r, par, centre(r))
        with torch.no_grad(): pv = torch.softmax(net(torch.tensor(r).permute(2, 0, 1)[None].float() / 255), 1)[0, 1].numpy()
        X.append(stats6(r[..., 1][od > 0]) + stats6(pv[r.mean(2) > 20]) + glcm_feats(raw))
    X = np.array(X); names = ["CNN (Adam only)", "CNN + OGO output layer"]

    def cv(X, y, seed):                     # 5-fold CV; scaling, Tanimoto selection, Adam and OGO are all fitted inside each training fold
        oof = np.zeros((2, len(y)))
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            predict, th0, th, _, _ = fit_model(X[tr], y[tr], seed * 10 + k); oof[0, te], oof[1, te] = predict(X[te], th0), predict(X[te], th)
        return oof
    runs = [cv(X, y, s_) for s_ in range(CV_REPEATS)]
    for i, n in enumerate(names):
        ms = [metrics(y, o[i]) for o in runs]
        print(f"[CLS] {n:24s}", {k: f"{np.mean([m[k] for m in ms]):.3f}+-{np.std([m[k] for m in ms]):.3f}" for k in ms[0]}, f"({CV_REPEATS} CV repeats)")
    y_shuf = np.random.default_rng(0).permutation(y)         # sanity check: with shuffled labels the AUC must fall to about 0.5
    print("[CLS] label-shuffle check, AUC =", metrics(y_shuf, cv(X, y_shuf, 0)[0])["auc"], "(about 0.5 expected; much higher would mean leakage)")
    plt.figure(figsize=(5, 5))
    for n, p in zip(names, runs[0]): fpr, tpr, _ = roc_curve(y, p); plt.plot(fpr, tpr, label=n)
    plt.plot([0, 1], [0, 1], "k:"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("out-of-fold ROC (repeat 0)"); plt.savefig(OUT / "cls_roc.png", dpi=110)
    predict, th0, th, hist, cnn = fit_model(X, y); torch.save({"cnn": cnn.state_dict(), "ogo_head": th}, OUT / "classifier.pt")


# ------------------------------------------------------------------ run everything that has data
if __name__ == "__main__":
    if IDRID.exists(): stage_od()
    else: print("skip OD stage: data/IDRiD not found")
    if DRIVE.exists(): stage_bvs()
    else: print("skip BVS stage: data/DRIVE not found")
    if JSIEC.exists() and (OUT / "resunet.pt").exists(): stage_cls()
    else: print("skip classification: need data/JSIEC and outputs/resunet.pt (run the BVS stage first)")