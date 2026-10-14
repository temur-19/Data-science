"""
Parallel hisoblash laboratoriya ishi - 8-variant (OpenMP g'oyasi, Python'da)

Variant: f(i) = i mod 7 + sqrt(i),  N = 3*10^7,  yig'indi = sum a_i,
         notekis yuk: i-iteratsiya ~ i * 1200 amal,  p = 1, 2, 4

Python'da GIL bor, shuning uchun OpenMP "oqimlari" o'rniga jarayonlar
(multiprocessing) ishlatiladi:
  - parallel for        -> iteratsiyalarni jarayonlarga bo'lish
  - reduction(+:sum)    -> har jarayon o'z qismiy yig'indisini qaytaradi, asosiy jarayon qo'shadi
  - reductionsiz        -> hamma jarayon bitta umumiy o'zgaruvchiga himoyasiz yozadi (race condition)
  - schedule(static)    -> iteratsiyalar p ta uzluksiz blokka bo'linadi
  - schedule(dynamic,c) -> c ta iteratsiyali bloklar bo'sh jarayonga navbat bilan beriladi

Ishga tushirish:   pip install numpy matplotlib
                   python lab_omp_v8.py
"""

import os
import sys
import time
import csv
import math
import multiprocessing as mp

import numpy as np

N = 30_000_000       # tekis yukli sikl uzunligi
M = 1000             # notekis (uchburchak) siklda iteratsiyalar soni
K = 1200             # i-iteratsiya ~ i*K amal
PS = [1, 2, 4]       # oqimlar (jarayonlar) soni
CHUNKS = [1, 10, 100]

_shared = None       # race-condition uchun umumiy o'zgaruvchi


def _init_shared(v):
    global _shared
    _shared = v


# ---------------------------------------------------------------- tekis yuk
def partial_sum(lo, hi):
    """sum_{i=lo}^{hi-1} (i mod 7 + sqrt(i))"""
    i = np.arange(lo, hi, dtype=np.float64)
    return float((np.mod(i, 7.0) + np.sqrt(i)).sum())


def w_reduction(args):
    lo, hi = args
    return partial_sum(lo, hi)           # qismiy yig'indi qaytadi -> reduction


def w_race(args):
    lo, hi, block = args
    for b in range(lo, hi, block):
        s = partial_sum(b, min(b + block, hi))
        _shared.value += s               # o'qi-qo'sh-yoz, hech qanday himoyasiz


def static_ranges(start, stop, p):
    """schedule(static): p ta uzluksiz blok"""
    n = stop - start
    base, extra = divmod(n, p)
    out, lo = [], start
    for k in range(p):
        hi = lo + base + (1 if k < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


# ---------------------------------------------------------------- notekis yuk
def heavy(i):
    """i-iteratsiyaning og'irligi i*K ga proporsional"""
    j = np.arange(i * K, dtype=np.int64)
    return float((j % 7).sum()) * 0.5


def heavy_range(args):
    lo, hi = args
    s = 0.0
    for i in range(lo, hi):
        s += heavy(i)
    return s


def run_pool(p, func, tasks, ordered_static=False):
    """Pool yaratib, vazifalarni bajaradi; (yig'indi, vaqt) qaytaradi.
    Pool vaqtga kirmaydi (isitish bilan)."""
    with mp.Pool(p) as pool:
        pool.map(abs, range(p), chunksize=1)          # jarayonlarni ishga tushirib qo'yish
        t0 = time.perf_counter()
        if ordered_static:
            res = pool.map(func, tasks, chunksize=1)
        else:
            res = list(pool.imap_unordered(func, tasks, chunksize=1))   # dynamic
        t = time.perf_counter() - t0
    return sum(res), t


def table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join("{:>%d}" % w for w in widths) + " |"
    print(line)
    print(fmt.format(*header))
    print(line)
    for r in rows:
        print(fmt.format(*r))
    print(line)


def main():
    results = {}

    # ---------------- 1. oqimlar soni
    cores = os.cpu_count()
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = cores
    print("=== 1. Oqimlar soni ===")
    print(f"Mantiqiy yadrolar (os.cpu_count)      : {cores}")
    print(f"Jarayonga ruxsat etilgan yadrolar     : {avail}")
    print(f"Sukut bo'yicha Pool() hajmi           : {cores}  (OpenMP'da omp_get_max_threads() ga mos)")
    print("Solishtirish: oqimlar soni yadrolar soniga teng -> har yadroga 1 oqim, ortig'i ortiqcha yuk.")

    # ---------------- 2. tekis yuk, parallel for, tezlanish
    print(f"\n=== 2. Tekis yukli sikl (N = {N:,}) ===")
    rows, T1 = [], None
    for p in PS:
        s, t = run_pool(p, w_reduction, static_ranges(1, N + 1, p), ordered_static=True)
        if p == 1:
            T1 = t
        rows.append([p, f"{t:.4f}", f"{T1 / t:.3f}", f"{T1 / t / p:.3f}", f"{s:.6f}"])
        results.setdefault("flat", []).append((p, t, T1 / t))
    table("Tekis yuk natijalari", ["p", "T, s", "S=T1/Tp", "E=S/p", "yig'indi"], rows)

    # ---------------- 3. reductionsiz (race) va reduction bilan
    print("\n=== 3. Reduksiya: bandsiz vs band bilan ===")
    BLOCK = 2000
    rows = []
    for p in PS:
        v = mp.RawValue("d", 0.0)                       # lock yo'q -> himoyasiz
        with mp.Pool(p, initializer=_init_shared, initargs=(v,)) as pool:
            tasks = [(lo, hi, BLOCK) for lo, hi in static_ranges(1, N + 1, p)]
            pool.map(w_race, tasks, chunksize=1)
        bad = v.value
        good, _ = run_pool(p, w_reduction, static_ranges(1, N + 1, p), ordered_static=True)
        rows.append([p, f"{bad:.6f}", f"{good:.6f}", f"{good - bad:.6f}"])
    table("Bandsiz (race) va reduction", ["p", "bandsiz", "reduction", "farq"], rows)
    print("Izoh: p=1 da farq yo'q; p>1 da bir vaqtda yozish tufayli ba'zi qo'shimchalar yo'qoladi,\n"
          "natija xato chiqadi va har ishga tushirishda turlicha bo'lishi mumkin.")

    # ---------------- 4. yig'indini oxirgi raqamlargacha solishtirish
    print("\n=== 4. Yig'indi (to'liq aniqlikda) ===")
    sums = {}
    for p in PS:
        s, _ = run_pool(p, w_reduction, static_ranges(1, N + 1, p), ordered_static=True)
        sums[p] = s
        print(f"p={p}  sum = {s!r}")
    ref = sums[1]
    for p in PS[1:]:
        print(f"p={p}  |sum_p - sum_1| = {abs(sums[p] - ref):.3e}")
    print("Sabab: suzuvchi nuqtali qo'shish assotsiativ emas ((a+b)+c != a+(b+c) yaxlitlash tufayli).\n"
          "Oqimlar soni o'zgarsa, qo'shish tartibi va qismiy yig'indilar o'zgaradi -> oxirgi raqamlar farq qiladi.")

    # ---------------- 5. uchburchak yuk, static + nazariy qiymat
    print(f"\n=== 5. Uchburchak yuk, schedule(static), M={M}, K={K} ===")
    rows, Ts1 = [], None
    for p in PS:
        _, t = run_pool(p, heavy_range, static_ranges(1, M + 1, p), ordered_static=True)
        if p == 1:
            Ts1 = t
        th = p * p / (2 * p - 1)
        rows.append([p, f"{t:.4f}", f"{Ts1 / t:.3f}", f"{th:.3f}"])
        results.setdefault("static", []).append((p, t, Ts1 / t))
    table("5.5-jadval: schedule(static)", ["p", "T, s", "S amalda", "S nazariy p^2/(2p-1)"], rows)

    # ---------------- 6. dynamic,1
    print("\n=== 6. Uchburchak yuk, schedule(dynamic, 1) ===")
    rows, Td1 = [], None
    for p in PS:
        tasks = [(i, i + 1) for i in range(1, M + 1)]
        _, t = run_pool(p, heavy_range, tasks)
        if p == 1:
            Td1 = t
        rows.append([p, f"{t:.4f}", f"{Td1 / t:.3f}", f"{Ts1 / t:.3f}"])
        results.setdefault("dynamic1", []).append((p, t, Td1 / t))
    table("5.6-jadval: schedule(dynamic,1)", ["p", "T, s", "S (o'z T1 ga nisbatan)", "S (static T1 ga nisbatan)"], rows)
    print("Xulosa: yuk notekis bo'lsa dynamic muvozanatni yaxshilaydi; yuk tekis bo'lsa static arzonroq\n"
          "(navbat boshqaruvi xarajati yo'q).")

    # ---------------- 7. blok hajmi
    print("\n=== 7. dynamic, blok hajmi 1 / 10 / 100 (p=4) ===")
    rows = []
    for c in CHUNKS:
        tasks = [(lo, min(lo + c, M + 1)) for lo in range(1, M + 1, c)]
        _, t = run_pool(4, heavy_range, tasks)
        rows.append([c, len(tasks), f"{t:.4f}", f"{Ts1 / t:.3f}"])
        results.setdefault("chunks", []).append((c, t, Ts1 / t))
    table("Blok hajmining ta'siri", ["chunk", "bloklar soni", "T, s", "S=T1/Tp"], rows)
    print("Izoh: chunk kichik -> yaxshi muvozanat, lekin navbatga murojaat ko'p (xarajat yuqori);\n"
          "chunk katta -> xarajat kam, lekin oxirgi og'ir bloklar tufayli muvozanat yomonlashadi.\n"
          "(5.3)-ifodadagi ikki haddan qaysi biri ustun kelishini jadvaldagi T dan aniqlang.")

    # ---------------- 8. CSV va diagramma
    with open("results_v8.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bo'lim", "p/chunk", "T_s", "S"])
        for name, data in results.items():
            for a, t, s in data:
                w.writerow([name, a, f"{t:.6f}", f"{s:.4f}"])
    print("\nNatijalar results_v8.csv fayliga yozildi.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.5))
        plt.plot(PS, PS, "k--", label="ideal S=p")
        for key, lab in (("flat", "tekis yuk"), ("static", "uchburchak, static"),
                         ("dynamic1", "uchburchak, dynamic,1")):
            plt.plot([x[0] for x in results[key]], [x[2] for x in results[key]], "o-", label=lab)
        plt.xlabel("p (jarayonlar soni)")
        plt.ylabel("Tezlanish S = T1/Tp")
        plt.xticks(PS)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title("Tezlanish diagrammasi (8-variant)")
        plt.tight_layout()
        plt.savefig("speedup_v8.png", dpi=150)
        print("Diagramma speedup_v8.png fayliga saqlandi.")
    except ImportError:
        print("matplotlib o'rnatilmagan: pip install matplotlib (diagramma chizish uchun).")


if __name__ == "__main__":
    mp.freeze_support()
    main()