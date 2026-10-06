# -*- coding: utf-8 -*-
"""GoFast — тест гипотезы про расстояние поездки и порог 3130 м.

Запуск: python3 gofast_distance_test.py
Требуется: pandas, numpy, scipy.
Результат: print в консоль + output/distance_test.txt

Постановка уточняется в ДВУХ версиях, потому что исходная формулировка
«расстояние одной поездки ≤ 3130 м у подписчиков» неоднозначна:
  V1. «Среднее расстояние за поездку НЕ превышает 3130 м» ->
      односторонний одновыборочный t-test: H0: μ ≤ 3130 против H1: μ > 3130.
  V2. «Доля поездок длиннее 3130 м НЕ превышает X %» ->
      биномиальный тест на пропорцию с CI (часто более честная версия).

ВАЖНО (оговорка в выводе): «оптимальность 3130 м с точки зрения износа» —
это БИЗНЕС-предположение, а не статистическая гипотеза. В данных нет индикаторов
износа/поломок, поэтому проверить «оптимальность» нельзя. Проверяем только
«превышает ли фактический показатель порог 3130 м».

Кластеризация: у одного пользователя много поездок -> помимо теста по поездкам
даём тест по пользовательским средним (одно значение на человека).
"""
import numpy as np
import pandas as pd
from scipy import stats
from gofast_stats_common import holm_adjust, load_data, OUT

ALPHA = 0.05
THRESHOLD = 3130.0  # м, порог из постановки
SHARE_REF = 0.50    # эталонная доля для V2 (доля «длинных» поездок, 50 %)
RNG = np.random.default_rng(7)

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


users, rides, _subs = load_data()
rides_m = rides.merge(users[["user_id", "subscription_type", "city"]], on="user_id", how="left")
rides_m["month"] = rides_m["date"].dt.to_period("M").astype(str)

log("=" * 72)
log(f"GoFast — ТЕСТ РАССТОЯНИЯ: порог {THRESHOLD:.0f} м")
log("=" * 72)

# ---------- QC ----------
log("\n### 0. КОНТРОЛЬ КАЧЕСТВА РАСПРЕДЕЛЕНИЯ distance ###")
q = rides.distance.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
log(f"min={rides.distance.min():.1f}, max={rides.distance.max():.1f}")
log(f"квантили: 0.1%={q[0.001]:.1f}, 1%={q[0.01]:.1f}, 25%={q[0.25]:.1f}, "
    f"медиана={q[0.5]:.1f}, 75%={q[0.75]:.1f}, 99%={q[0.99]:.1f}, 99.9%={q[0.999]:.1f}")
above = rides.distance > THRESHOLD
log(f"Поездок > {THRESHOLD:.0f} м: {above.sum()} из {len(rides)} ({above.mean():.2%}) "
    f"— хвост распределения плотный, порог лежит почти по центру.")
log(f"Поездок distance <= 0: {(rides.distance <= 0).sum()}")


def v1_one_sample_t(dist, label, alternative="greater"):
    """Одновыборочный t-test H0: μ <= THRESHOLD против H1: μ > THRESHOLD."""
    t = stats.ttest_1samp(dist, THRESHOLD, alternative=alternative)
    se = dist.std(ddof=1) / np.sqrt(len(dist))
    df = len(dist) - 1
    tcrit = stats.t.ppf(1 - ALPHA / 2, df)
    ci = (dist.mean() - tcrit * se, dist.mean() + tcrit * se)
    log(f"[{label}] n={len(dist)}, mean={dist.mean():.1f} м, median={np.median(dist):.1f} м")
    log(f"    H0: μ ≤ {THRESHOLD:.0f}; H1: μ > {THRESHOLD}. t={t.statistic:.2f}, df={df}, "
        f"p(one-sided)={t.pvalue:.3e}")
    log(f"    Среднее {dist.mean():.1f} м против порога {THRESHOLD:.0f} м "
        f"(разница {dist.mean() - THRESHOLD:+.1f} м), 95% CI среднего "
        f"[{ci[0]:.1f}; {ci[1]:.1f}]")
    verdict = ("Среднее ЗНАЧИМО выше порога -> гипотеза «≤ 3130 м» ОТВЕРГАЕТСЯ"
               if t.pvalue < ALPHA else "Нет оснований отвергнуть μ ≤ 3130 (среднее не выше порога)")
    log(f"    Вывод: {verdict}")
    return t.pvalue


def v2_binomial(dist, label, p0=SHARE_REF):
    """Биномиальный тест доли поездок > THRESHOLD против p0."""
    k = int((dist > THRESHOLD).sum())
    n = len(dist)
    bt = stats.binomtest(k, n, p0, alternative="two-sided")
    ci = bt.proportion_ci(confidence_level=1 - ALPHA, method="wilson")
    log(f"[{label}] доля > {THRESHOLD:.0f} м = {k}/{n} = {k/n:.2%}, "
        f"H0: p = {p0:.0%}")
    log(f"    Биномиальный тест: p={bt.pvalue:.3e}; 95% CI доли (Wilson) "
        f"[{ci.low:.2%}; {ci.high:.2%}]")
    return k, n, bt.pvalue, ci


log("\n### V1. ОДНОВЫБОРОЧНЫЙ T-TEST: среднее расстояние ≤ 3130 м? ###")
groups = {
    "ultra": rides_m.loc[rides_m.subscription_type == "ultra", "distance"].to_numpy(),
    "free": rides_m.loc[rides_m.subscription_type == "free", "distance"].to_numpy(),
    "все": rides_m["distance"].to_numpy(),
}
v1_p = {}
for label, arr in groups.items():
    v1_p[label] = v1_one_sample_t(arr, label)
    log("")

# по пользовательским средним (учёт кластеризации)
log("--- V1 по ПОЛЬЗОВАТЕЛЬСКИМ средним (кластеризация учтена) ---")
for label in ["ultra", "free"]:
    sub = rides_m[rides_m.subscription_type == label]
    ud = sub.groupby("user_id")["distance"].mean().to_numpy()
    p = v1_one_sample_t(ud, f"{label}/user")
    log("")

log("### V2. БИНОМИАЛЬНЫЙ ТЕСТ ДОЛИ поездок > 3130 м ###")
v2_all = v2_binomial(rides_m["distance"].to_numpy(), "все поездки")
log(f"    Справочная доля p0={SHARE_REF:.0%} взята как естественная точка «половина». "
    "Можно заменить на бизнес-норму X %.")
for label, arr in groups.items():
    k, n, p, ci = v2_binomial(arr, label)
    log("")

log("### Стратификация по городам (V1, Holm) ###")
crows, cp = [], []
for c, g in rides_m.groupby("city"):
    d = g["distance"].to_numpy()
    t = stats.ttest_1samp(d, THRESHOLD, alternative="greater")
    crows.append((c, d.mean(), len(d), t.statistic, t.pvalue))
    cp.append(t.pvalue)
cad = holm_adjust(cp)
for (c, m, n, ts, p), pa in zip(crows, cad):
    log(f"{c}: mean={m:.1f} м (n={n}), t={ts:.2f}, p={p:.3e}, p_Holm={pa:.3e}")

log("### Стратификация по месяцам (V1, Holm) ###")
mrows, mp = [], []
for mon, g in rides_m.groupby("month"):
    d = g["distance"].to_numpy()
    t = stats.ttest_1samp(d, THRESHOLD, alternative="greater")
    mrows.append((mon, d.mean(), len(d), t.statistic, t.pvalue))
    mp.append(t.pvalue)
mad = holm_adjust(mp)
for (mon, m, n, ts, p), pa in zip(mrows, mad):
    log(f"{mon}: mean={m:.1f} м (n={n}), t={ts:.2f}, p={p:.3e}, p_Holm={pa:.3e}")

log("\n" + "=" * 72)
log("ВЫВОД")
log("=" * 72)
for label, arr in groups.items():
    log(f"- «{label}»: среднее {arr.mean():.1f} м "
        f"{'ВЫШЕ' if arr.mean() > THRESHOLD else 'ниже'} порога {THRESHOLD:.0f} м "
        f"(p(V1)={v1_p[label]:.3e})")
log(f"- Доля поездок > {THRESHOLD:.0f} м: {v2_all[0]}/{v2_all[1]} = "
    f"{v2_all[0]/v2_all[1]:.2%} (95% CI [{v2_all[3].low:.2%}; {v2_all[3].high:.2%}])")
log("- Формально подтверждается только версия «среднее/доля относительно порога».")
log("- «Оптимальность 3130 м с точки зрения износа» — бизнес-предположение, "
    "НЕ статистическая гипотеза: индикаторов износа/поломок в данных нет.")

(OUT / "distance_test.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nОтчет сохранён: {OUT / 'distance_test.txt'}")
