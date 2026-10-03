"""Verify the statistics and PRNG that CUS Reader 0.8.0 will implement in JS.

Each block computes the quantity the way the JS will, then checks it against a
reference implementation (scipy / sklearn / statsmodels). Test vectors are
emitted to handoff/test_vectors.json for embedding in the browser self-test.
"""
import json, math, os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.stats.proportion import proportion_confint

os.makedirs("handoff", exist_ok=True)
report = []
vectors = {}


def chk(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if want is not None else got is None
    report.append((name, got, want, ok))
    return ok


# ---------------------------------------------------------------- Wilson CI
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)

for k, n in [(4, 4), (8, 10), (0, 5), (1, 2), (16, 20), (50, 50), (3, 7)]:
    lo, hi = wilson(k, n)
    rlo, rhi = proportion_confint(k, n, alpha=0.05, method="wilson")
    chk(f"wilson lo {k}/{n}", lo, float(rlo), 1e-12)
    chk(f"wilson hi {k}/{n}", hi, float(rhi), 1e-12)

# smallest n with perfect performance that clears a 0.80 Wilson lower bound
n_needed = next(n for n in range(1, 400) if wilson(n, n)[0] >= 0.80)
report.append(("n for perfect-score Wilson LB>=0.80", n_needed, None, True))
# and what 0.7.0's minimum gate (2 pos / 2 neg, point estimate 1.0) actually proves
lb_2of2 = wilson(2, 2)[0]
report.append(("Wilson LB at 2/2 correct", round(lb_2of2, 4), None, True))
# chance that a coin-flip classifier clears 0.8 sens AND 0.8 spec at 2 pos/2 neg
p_coin = (0.5 ** 2) ** 2
report.append(("P(coin flip clears 0.7.0 gate, 2pos/2neg)", p_coin, None, True))

# ------------------------------------------------------------------- AUROC
def auroc(y, s):
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):           # midranks for ties
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        mid = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = mid
        i = j + 1
    rsum = sum(ranks[i] for i in pos)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

rng = np.random.default_rng(7)
auc_cases = [
    ([1, 0, 1, 0, 1], [0.9, 0.1, 0.8, 0.4, 0.6]),
    ([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]),          # all ties -> 0.5
    ([1, 0, 0, 1, 1, 0], [0.2, 0.2, 0.9, 0.9, 0.5, 0.5]),  # ties straddling
]
y_r = (rng.random(40) < 0.4).astype(int).tolist()
s_r = np.round(rng.random(40), 4).tolist()
auc_cases.append((y_r, s_r))
for idx, (y, s) in enumerate(auc_cases):
    chk(f"auroc case{idx}", auroc(y, s), float(roc_auc_score(y, s)), 1e-12)

# ------------------------------------------------------------------- Brier
def brier(y, s):
    return sum((si - yi) ** 2 for yi, si in zip(y, s)) / len(y)

chk("brier", brier(y_r, s_r), float(brier_score_loss(y_r, s_r)), 1e-12)

# ------------------------------------------- Youden J threshold on a fold
def youden(y, s):
    """Threshold maximising sensitivity+specificity-1; ties resolved toward 0.5."""
    best = (None, -2.0)
    for t in sorted(set(s)) + [1.000001]:
        tp = sum(1 for yi, si in zip(y, s) if yi == 1 and si >= t)
        fn = sum(1 for yi, si in zip(y, s) if yi == 1 and si < t)
        tn = sum(1 for yi, si in zip(y, s) if yi == 0 and si < t)
        fp = sum(1 for yi, si in zip(y, s) if yi == 0 and si >= t)
        if tp + fn == 0 or tn + fp == 0:
            continue
        j = tp / (tp + fn) + tn / (tn + fp) - 1
        if j > best[1] + 1e-12 or (abs(j - best[1]) <= 1e-12 and best[0] is not None
                                   and abs(t - 0.5) < abs(best[0] - 0.5)):
            best = (t, j)
    return best

t_y, j_y = youden([1, 1, 0, 0, 1, 0], [0.8, 0.6, 0.55, 0.2, 0.9, 0.3])
report.append(("youden threshold", t_y, 0.6, abs(t_y - 0.6) < 1e-12))
report.append(("youden J", round(j_y, 6), 1.0, abs(j_y - 1.0) < 1e-12))

# ------------------------------------------------- mulberry32 PRNG (JS-exact)
M32 = 0xFFFFFFFF
def imul(a, b):
    r = (a & M32) * (b & M32) & M32
    return r - 0x100000000 if r & 0x80000000 else r

def mulberry32(seed):
    a = seed & M32
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = imul(a ^ (a >> 15), 1 | a) & M32
        t = (imul(t ^ (t >> 7), 61 | t) ^ t) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return nxt

r1 = mulberry32(42)
prng_seq = [round(r1(), 12) for _ in range(6)]
report.append(("mulberry32(42)[0:3]", prng_seq[:3], None, True))

def shuffle(items, rand):
    """Fisher-Yates, the fix for 0.7.0's sort(()=>Math.random()-.5) comparator."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = int(rand() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out

shuf = shuffle(list(range(10)), mulberry32(7))
report.append(("fisher-yates seed7 of range(10)", shuf, None, True))
assert sorted(shuf) == list(range(10))

# Note: 0.7.0 shuffles with sort(()=>Math.random()-.5). A random comparator is
# not a uniform shuffle -- the resulting distribution depends on V8's TimSort
# internals, which cannot be faithfully emulated here, so no bias magnitude is
# claimed. Fisher-Yates below is uniform by construction and replaces it.

# ------------------------------------------------ cluster bootstrap on delta
def balacc(y, s, t):
    tp = sum(1 for yi, si in zip(y, s) if yi == 1 and si >= t)
    fn = sum(1 for yi, si in zip(y, s) if yi == 1 and si < t)
    tn = sum(1 for yi, si in zip(y, s) if yi == 0 and si < t)
    fp = sum(1 for yi, si in zip(y, s) if yi == 0 and si >= t)
    if tp + fn == 0 or tn + fp == 0:
        return None
    return (tp / (tp + fn) + tn / (tn + fp)) / 2

def cluster_bootstrap_delta(infants, y, sa, sb, reps, rand):
    groups = {}
    for i, g in enumerate(infants):
        groups.setdefault(g, []).append(i)
        keys = list(groups)
    keys = sorted(groups)
    deltas = []
    for _ in range(reps):
        idx = []
        for _ in range(len(keys)):
            idx.extend(groups[keys[int(rand() * len(keys))]])
        ya = [y[i] for i in idx]
        a = balacc(ya, [sa[i] for i in idx], 0.5)
        b = balacc(ya, [sb[i] for i in idx], 0.5)
        if a is not None and b is not None:
            deltas.append(a - b)
    deltas.sort()
    if len(deltas) < 100:
        return None
    return (deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))], len(deltas))

inf = [f"I{i//2}" for i in range(24)]
yy = [i % 2 for i in range(24)]
cand = [0.9 if v else 0.1 for v in yy]                      # near-perfect
base = [0.9 if (v and i % 4 == 0) else 0.45 for i, v in enumerate(yy)]  # weak
bs = cluster_bootstrap_delta(inf, yy, cand, base, 2000, mulberry32(11))
report.append(("bootstrap delta 95% CI (strong vs weak)",
               (round(bs[0], 4), round(bs[1], 4)), "LB>0", bs[0] > 0))
bs_same = cluster_bootstrap_delta(inf, yy, cand, cand, 2000, mulberry32(11))
report.append(("bootstrap delta CI (model vs itself)",
               (round(bs_same[0], 4), round(bs_same[1], 4)), "contains 0",
               bs_same[0] <= 0 <= bs_same[1]))

# ------------------------------------------------------ conv stack shapes
def stack(edge, filters=(12, 24, 32, 48)):
    """padding='same' convs with a stride-2 first conv, maxpool 2 between."""
    s, params, trace = edge, 0, []
    s = math.ceil(s / 2); params += 5 * 5 * 1 * filters[0] + filters[0]
    trace.append(("conv1 k5 s2 same", s, filters[0]))
    prev = filters[0]
    for i, f in enumerate(filters[1:], start=2):
        s = s // 2; trace.append((f"pool{i-1}", s, prev))
        params += 3 * 3 * prev * f + f
        trace.append((f"conv{i} k3 same", s, f))
        prev = f
    return s, prev, params, trace

shapes = {}
for edge in (64, 96, 128):
    s, d, p, tr = stack(edge)
    shapes[edge] = dict(final_map=s, channels=d, encoder_params=p)
    assert s >= 2, f"spatial map collapsed at edge={edge}"
D = 2 * shapes[64]["channels"]                     # avg-pool + max-pool concat
A = 24
head_in = 2 * D                                    # attention-pooled + frame-max
HEADS_080 = 22   # 15 hemisphere-specific + 7 side-agnostic
HEADS_070 = 15
total = (shapes[64]["encoder_params"]
         + D * A + A + A * 1 + 1
         + head_in * 64 + 64 + 64 * HEADS_080 + HEADS_080)
old = (5 * 5 * 1 * 8 + 8) + (3 * 3 * 8 * 16 + 16) + (32 * 24 + 24) + (24 * HEADS_070 + HEADS_070)
report.append(("frame embedding dim D", D, 96, D == 96))
report.append(("head input dim", head_in, 192, head_in == 192))
report.append(("0.8.0 trainable params", total, None, True))
report.append(("0.7.0 trainable params", old, None, True))

# ------------------------------------------- NIfTI windowing: the 0.7.0 bug
def old_window(v, mn, mx):
    """0.7.0 decodeNifti, verbatim: values already inside 0..255 are passed
    through unscaled, otherwise divided by max(range, 1). Both lose contrast."""
    if mn >= 0 and mx <= 255:
        return np.asarray(v, float)
    return (np.asarray(v, float) - mn) / max(mx - mn, 1) * 255

def new_window(vals, lo_pct=1.0, hi_pct=99.0):
    a = np.sort(np.asarray(vals, float))
    lo = a[max(0, min(len(a) - 1, int(lo_pct / 100 * (len(a) - 1))))]
    hi = a[max(0, min(len(a) - 1, int(hi_pct / 100 * (len(a) - 1))))]
    if hi <= lo:
        lo, hi = a[0], a[-1]
    if hi <= lo:
        return np.zeros(len(a)), lo, hi
    return np.clip((np.asarray(vals, float) - lo) / (hi - lo), 0, 1) * 255, lo, hi

float_vol = np.linspace(0.0, 0.3, 4096)              # float NIfTI with range < 1
old_span = np.ptp(old_window(float_vol, 0.0, 0.3))   # pass-through branch
new_span = np.ptp(new_window(float_vol)[0])
report.append(("float NIfTI range 0-0.3: 0.7.0 span (of 255, pass-through)", round(float(old_span), 3),
               255.0, False))
report.append(("float NIfTI range 0-0.3: 0.8.0 span (of 255)", round(float(new_span), 3),
               255.0, abs(new_span - 255) < 1e-6))
lowdr = np.linspace(0, 12, 4096)                     # low-dynamic-range int volume
report.append(("int 0-12 volume: 0.7.0 span (of 255, pass-through)",
               round(float(np.ptp(lowdr)), 3), 255.0, False))
report.append(("int 0-12 volume: 0.8.0 span", round(float(np.ptp(new_window(lowdr)[0])), 3),
               255.0, abs(np.ptp(new_window(lowdr)[0]) - 255) < 1e-6))

# ------------------------------------------------- letterbox resize geometry
def letterbox(w, h, edge):
    scale = min(edge / w, edge / h)
    dw, dh = round(w * scale), round(h * scale)
    return dw, dh, (edge - dw) // 2, (edge - dh) // 2

for (w, h) in [(800, 600), (1024, 768), (640, 480), (512, 512), (720, 540), (1280, 720)]:
    dw, dh, ox, oy = letterbox(w, h, 96)
    src_ar, dst_ar = w / h, dw / dh
    chk(f"letterbox aspect {w}x{h}", dst_ar, src_ar, 0.01)
    # the 0.7.0 stretch-to-square distortion, for comparison
report.append(("0.7.0 aspect distortion 1280x720 -> square",
               round((96 / 96) / (1280 / 720), 4), 1.0, False))

# ---------------------------------------------------------------- emit
vectors = {
    "wilson": [{"k": k, "n": n, "lo": wilson(k, n)[0], "hi": wilson(k, n)[1]}
               for k, n in [(4, 4), (8, 10), (0, 5), (1, 2), (16, 20), (3, 7)]],
    "auroc": [{"y": y, "s": s, "auc": auroc(y, s)} for y, s in auc_cases],
    "brier": {"y": y_r, "s": s_r, "value": brier(y_r, s_r)},
    "youden": {"y": [1, 1, 0, 0, 1, 0], "s": [0.8, 0.6, 0.55, 0.2, 0.9, 0.3],
               "threshold": t_y},
    "prng": {"seed": 42, "first6": prng_seq},
    "shuffle": {"seed": 7, "n": 10, "result": shuf},
    "nWilson80": n_needed,
    "shapes": {str(k): v for k, v in shapes.items()},
    "params": {"v080": total, "v070": old},
    "letterbox": [{"w": w, "h": h, "out": letterbox(w, h, 96)}
                  for w, h in [(800, 600), (1280, 720), (512, 512)]],
}
json.dump(vectors, open("handoff/test_vectors.json", "w"), indent=1)

bad = [r for r in report if r[3] is False and r[2] is not None]
for name, got, want, ok in report:
    print(f"{'ok ' if ok else '!! '}{name}: {got}" + (f"  [ref {want}]" if want is not None else ""))
print(f"\nchecks={len(report)}  intentional-mismatch(0.7.0 behaviour)={len(bad)}")
