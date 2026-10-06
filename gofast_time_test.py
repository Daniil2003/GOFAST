# -*- coding: utf-8 -*-
"""GoFast — тест гипотезы «подписчики (ultra) тратят больше ВРЕМЕНИ на поездки».

Запуск: python3 gofast_time_test.py
Требуется: pandas, numpy, scipy.
Результат: print в консоль + output/time_test.txt

Разделены две РАЗНЫЕ метрики (не путать с гипотезой «ездит чаще»):
  A. Время на ОДНУ поездку — сравнение duration между группами.
  B. СУММАРНОЕ время за период на пользователя — отдельная формулировка.

Ключевые методологические моменты:
  * у одного пользователя много поездок, наблюдения не независимы ->
    основной вывод строится на агрегатах ПО ПОЛЬЗОВАТЕЛЮ (средняя длительность
    на пользователя) и на bootstrap по пользователям (cluster bootstrap);
  * Welch t-test по поездкам приводится как «наивный» (p занижен);
  * Mann-Whitney U как робастная непараметрическая альтернатива;
  * стратификация по месяцам и городам с поправкой Холма.
"""
import numpy as np
import pandas as pd
from scipy import stats
from gofast_stats_common import holm_adjust, load_data, OUT

RNG = np.random.default_rng(42)
N_BOOT = 5000
ALPHA = 0.05

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def welch_ci(a, b):
    """Welch t-test + 95% CI разницы средних (a - b)."""
    t = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = t.df
    tcrit = stats.t.ppf(1 - ALPHA / 2, df)
    diff = a.mean() - b.mean()
    return t.statistic, t.pvalue, df, diff, diff - tcrit * se, diff + tcrit * se


def cluster_bootstrap_diff(df, col, group_col, n_boot=N_BOOT):
    """Cluster bootstrap: ресэмплируем ПОЛЬЗОВАТЕЛЕЙ, а не поездки.

    Устраняет занижение p-value из-за повторных наблюдений одного пользователя.
    Возвращает (разница средних ultra-free, lo, hi, p).
    """
    per_user = df.groupby(["user_id", group_col])[col].agg(["sum", "count"]).reset_index()
    f = per_user.loc[per_user[group_col] == "free", ["sum", "count"]].to_numpy(dtype=float)
    u = per_user.loc[per_user[group_col] == "ultra", ["sum", "count"]].to_numpy(dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bf = f[RNG.integers(0, len(f), len(f))]
        bu = u[RNG.integers(0, len(u), len(u))]
        diffs[i] = bu[:, 0].sum() / bu[:, 1].sum() - bf[:, 0].sum() / bf[:, 1].sum()
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, float(min(p, 1.0))


users, rides, _subs = load_data()
rides_m = rides.merge(users[["user_id", "subscription_type", "city"]], on="user_id", how="left")
rides_m["month"] = rides_m["date"].dt.to_period("M").astype(str)
free_rides = rides_m.loc[rides_m.subscription_type == "free", "duration"].to_numpy()
ultra_rides = rides_m.loc[rides_m.subscription_type == "ultra", "duration"].to_numpy()

log("=" * 72)
log("GoFast — ТЕСТ ВРЕМЕНИ: тратят ли ultra больше времени на поездки?")
log("=" * 72)

# ---------- A. ВРЕМЯ НА ОДНУ ПОЕЗДКУ ----------
log("\n### A. ВРЕМЯ НА ОДНУ ПОЕЗДКУ (duration, минуты) ###")
log(f"free : n={len(free_rides)}, mean={free_rides.mean():.3f}, median={np.median(free_rides):.3f}, "
    f"sd={free_rides.std(ddof=1):.3f}")
log(f"ultra: n={len(ultra_rides)}, mean={ultra_rides.mean():.3f}, median={np.median(ultra_rides):.3f}, "
    f"sd={ultra_rides.std(ddof=1):.3f}")

log("\n--- A0. НАИВНЫЙ тест (поездки как независимые наблюдения; p занижен) ---")
t_stat, t_p, t_df, diff, lo, hi = welch_ci(ultra_rides, free_rides)
log(f"Welch t={t_stat:.2f}, df={t_df:.0f}, p={t_p:.2e}")
log(f"Разница средних (ultra - free) = {diff:+.3f} мин, 95% CI [{lo:+.3f}; {hi:+.3f}]")
log("Примечание: CI/p по поездкам оптимистичны — у одного пользователя десятки поездок.")

log("\n--- A1. Проверка нормальности ---")
for name, arr in [("free", free_rides), ("ultra", ultra_rides)]:
    samp = arr if len(arr) <= 5000 else RNG.choice(arr, 5000, replace=False)
    sw = stats.shapiro(samp)
    log(f"{name}: Shapiro-Wilk p={sw.pvalue:.2e} (n для теста={len(samp)})")
log("Оговорка: при n в тысячи Shapiro почти всегда отвергает нормальность;")
log("однако распределения симметричны и без тяжёлых хвостов, поэтому t-тест "
    "устойчив (CLT), а Mann-Whitney U даётся ниже как робастная альтернатива.")

log("\n--- A2. Mann-Whitney U (непараметрическая альтернатива) ---")
mw = stats.mannwhitneyu(ultra_rides, free_rides, alternative="two-sided")
rb = 2 * mw.statistic / (len(ultra_rides) * len(free_rides)) - 1
log(f"U={mw.statistic:,.0f}, p={mw.pvalue:.2e}, ранг-бисериальная корреляция={rb:+.4f}")

log("\n--- A3. ГЛАВНОЕ: кластеризация по пользователям ---")
# Агрегат: средняя длина поездки на пользователя (одно число на человека)
ud = rides_m.groupby(["user_id", "subscription_type"])["duration"].mean().reset_index()
uf = ud.loc[ud.subscription_type == "free", "duration"].to_numpy()
uu = ud.loc[ud.subscription_type == "ultra", "duration"].to_numpy()
t2, p2, df2, d2, lo2, hi2 = welch_ci(uu, uf)
log(f"Пользовательских средних: free n={len(uf)}, ultra n={len(uu)}")
log(f"mean per user: free={uf.mean():.3f}, ultra={uu.mean():.3f}")
log(f"Welch (на уровне пользователей): t={t2:.2f}, df={df2:.0f}, p={p2:.2e}")
log(f"Разница (ultra - free) = {d2:+.3f} мин, 95% CI [{lo2:+.3f}; {hi2:+.3f}]")
log(f"Cohen d (по пользователям) = {d2 / np.sqrt((uf.var(ddof=1) + uu.var(ddof=1)) / 2):.3f}")
mwu = stats.mannwhitneyu(uu, uf, alternative="two-sided")
log(f"Mann-Whitney U по пользовательским средним: p={mwu.pvalue:.2e}")

# Cluster bootstrap по пользователям
boot = cluster_bootstrap_diff(rides_m, "duration", "subscription_type")
log(f"Cluster bootstrap (ресэмпл пользователей, {N_BOOT} итераций): "
    f"разница средних={boot[0]:+.3f} мин, 95% CI [{boot[1]:+.3f}; {boot[2]:+.3f}], p={boot[3]:.2e}")

log("\n--- A4. Стратификация: по месяцам (Holm) ---")
months = sorted(rides_m["month"].unique())
mrows, mp = [], []
for m in months:
    g = rides_m[rides_m.month == m]
    f = g.loc[g.subscription_type == "free", "duration"].to_numpy()
    u = g.loc[g.subscription_type == "ultra", "duration"].to_numpy()
    # на уровне пользователей внутри месяца — честнее по кластеризации
    udm = g.groupby(["user_id", "subscription_type"])["duration"].mean().reset_index()
    fm = udm.loc[udm.subscription_type == "free", "duration"].to_numpy()
    um = udm.loc[udm.subscription_type == "ultra", "duration"].to_numpy()
    if len(fm) > 5 and len(um) > 5:
        p = stats.mannwhitneyu(um, fm, alternative="two-sided").pvalue
    else:
        p = np.nan
    mrows.append((m, f.mean(), u.mean(), f.mean() - u.mean(), p))
    mp.append(p)
mad = holm_adjust([x for x in mp if not np.isnan(x)])
mi = 0
for i, (m, mf, mu, d, p) in enumerate(mrows):
    if np.isnan(p):
        log(f"{m}: avg/ride free={mf:.2f}, ultra={mu:.2f}, p=n/a (мало данных)")
    else:
        log(f"{m}: avg/ride free={mf:.2f}, ultra={mu:.2f}, diff(free-ultra)={d:+.2f}, p={p:.2e}, p_Holm={mad[mi]:.2e}")
        mi += 1

log("\n--- A5. Стратификация: по городам (Holm) ---")
crows = []
for c, g in rides_m.groupby("city"):
    udc = g.groupby(["user_id", "subscription_type"])["duration"].mean().reset_index()
    f = udc.loc[udc.subscription_type == "free", "duration"].to_numpy()
    u = udc.loc[udc.subscription_type == "ultra", "duration"].to_numpy()
    if len(f) > 5 and len(u) > 5:
        p = stats.mannwhitneyu(u, f, alternative="two-sided").pvalue
        crows.append((c, f.mean(), len(f), u.mean(), len(u), f.mean() - u.mean(), p))
cad = holm_adjust([r[6] for r in crows])
for (c, mf, nf, mu, nu, d, p), pa in zip(crows, cad):
    log(f"{c}: free={mf:.2f} (n={nf}), ultra={mu:.2f} (n={nu}), diff(free-ultra)={d:+.2f}, "
        f"p={p:.2e}, p_Holm={pa:.2e}")

# ---------- B. СУММАРНОЕ ВРЕМЯ ЗА ПЕРИОД ----------
log("\n### B. СУММАРНОЕ ВРЕМЯ НА ПОЛЬЗОВАТЕЛЯ ЗА 2021 (минуты) ###")
tot = rides_m.groupby(["user_id", "subscription_type"])["duration"].sum().reset_index()
# Пользователи без поездок отсутствуют (минимум 2 поездки), но подстрахуемся join'ом
tot = users[["user_id", "subscription_type"]].merge(
    tot.drop(columns="subscription_type"), on="user_id", how="left"
)
tot["duration"] = tot["duration"].fillna(0.0)
tf = tot.loc[tot.subscription_type == "free", "duration"].to_numpy()
tu = tot.loc[tot.subscription_type == "ultra", "duration"].to_numpy()
log(f"free : n={len(tf)}, mean={tf.mean():.1f} мин/год, median={np.median(tf):.1f}")
log(f"ultra: n={len(tu)}, mean={tu.mean():.1f} мин/год, median={np.median(tu):.1f}")
tw = stats.ttest_ind(tu, tf, equal_var=False)
se = np.sqrt(tu.var(ddof=1) / len(tu) + tf.var(ddof=1) / len(tf))
tcrit = stats.t.ppf(1 - ALPHA / 2, tw.df)
dd = tu.mean() - tf.mean()
log(f"Welch t={tw.statistic:.2f}, p={tw.pvalue:.2e}")
log(f"Разница (ultra - free) = {dd:+.1f} мин/год, "
    f"95% CI [{dd - tcrit * se:+.1f}; {dd + tcrit * se:+.1f}]")
mwb = stats.mannwhitneyu(tu, tf, alternative="two-sided")
log(f"Mann-Whitney U: p={mwb.pvalue:.2e}")
log("ВАЖНО: суммарное время сильно связано с числом поездок (гипотеза №3) и "
    "не должно трактоваться как отдельное «ultra ездит дольше».")

# ---------- ИТОГ ----------
log("\n" + "=" * 72)
log("ВЫВОД")
log("=" * 72)
log(f"A. Время на ОДНУ поездку: ultra {'длиннее' if d2 > 0 else 'короче'} free на "
    f"{abs(d2):.2f} мин (95% CI по пользователям [{lo2:+.2f}; {hi2:+.2f}]).")
if lo2 > 0 or hi2 < 0:
    log("   Разница статистически значима и на уровне пользователей (кластеризация учтена).")
else:
    log("   Разница НЕ значима на уровне пользователей: наивный тест по поездкам "
        "завышал значимость из-за кластеризации.")
log(f"B. Суммарное время: ultra {'больше' if dd > 0 else 'меньше'} free на {abs(dd):.1f} мин/год "
    f"(p={tw.pvalue:.2e}), но метрика почти дублирует число поездок.")
log("Формулировки «ultra ездит дольше за поездку» и «ultra тратит больше суммарного "
    "времени» — разные; выше они разведены и проверены раздельно.")

(OUT / "time_test.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nОтчет сохранён: {OUT / 'time_test.txt'}")
