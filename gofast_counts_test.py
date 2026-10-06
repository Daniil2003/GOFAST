"""
GoFast — тест counts: n поездок / пользователь (free vs ultra).
Запуск: python3 gofast_counts_test.py
Требуется: pandas, numpy, scipy (statsmodels — опционально для NB GLM).
Результат: print в консоль + output/counts_test.txt

Метод:
  1. Основной тест — Mann-Whitney U (counts дискретны, распределение ненормально).
  2. Чувствительность — Welch t-test + Cohen's d.
  3. Poisson-тест НЕ применяется при сверхдисперсии (var/mean >> 1);
     вместо него — Negative Binomial (IRR) либо отношение средних с CI.
  4. Контроль сезонности — стратификация по месяцам и городам.
  5. Дата релиза — RELEASE_DATE: если известна, включается DiD pre/post блок;
     если None — честное кросс-секционное сравнение без причинного вывода.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from gofast_stats_common import dispersion_ratio, holm_adjust, load_data, OUT

# --- Фиксация даты релиза: поставьте 'YYYY-MM-DD', когда станет известна ---
RELEASE_DATE = None  # например: "2021-06-01"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


users, rides, _subs = load_data()

rpu_counts = rides.groupby("user_id").size().reset_index(name="n")
# Пользователи без поездок в данных отсутствуют (минимум 2 поездки на человека),
# поэтому left-join — защита от будущих выгрузок, а не признак реальных нулей.
rpu = (users[["user_id", "subscription_type", "city"]].merge(rpu_counts, on="user_id", how="left"))
rpu["n"] = rpu["n"].fillna(0).astype(int)
free = rpu.loc[rpu.subscription_type == "free", "n"].to_numpy()
ultra = rpu.loc[rpu.subscription_type == "ultra", "n"].to_numpy()

log("=== DESCRIPTIVES (n поездок/польз за 2021) ===")
log(f"free : n={len(free)}, mean={free.mean():.2f}, median={np.median(free):.0f}, "
    f"var/mean={dispersion_ratio(free):.2f}")
log(f"ultra: n={len(ultra)}, mean={ultra.mean():.2f}, median={np.median(ultra):.0f}, "
    f"var/mean={dispersion_ratio(ultra):.2f}")
log(f"Период данных: {rides['date'].min().date()} — {rides['date'].max().date()}")
log(f"Пользователей без поездок: {len(set(users.user_id) - set(rides.user_id))}")
log("Примечание: нулевых наблюдений нет (у каждого ≥2 поездки), поэтому средние "
    "смещены вверх относительно всей клиентской базы; на сравнение тарифов это "
    "не влияет — правило одинаково для обеих групп.")

log("\n=== NORMALITY (Shapiro-Wilk) ===")
log(f"free p={stats.shapiro(free).pvalue:.2e}, ultra p={stats.shapiro(ultra).pvalue:.2e} "
    "-> ненормально, основной тест непараметрический")

log("\n=== 1. MANN-WHITNEY U (основной) ===")
mw = stats.mannwhitneyu(free, ultra, alternative="two-sided")
log(f"U={mw.statistic:.0f}, p={mw.pvalue:.2e}")
log(f"Медианная разница: {np.median(free) - np.median(ultra):+.0f} поездок "
    f"(median free={np.median(free):.0f}, ultra={np.median(ultra):.0f})")
# Рангово-бисериальная корреляция как effect size для MWU
n1, n2 = len(free), len(ultra)
rank_biserial = 2 * mw.statistic / (n1 * n2) - 1
log(f"Рангово-бисериальная корреляция: {rank_biserial:+.3f} (знак + = free выше)")

log("\n=== 2. WELCH T-TEST (чувствительность) ===")
t = stats.ttest_ind(free, ultra, equal_var=False)
d = (free.mean() - ultra.mean()) / np.sqrt((free.var(ddof=1) + ultra.var(ddof=1)) / 2)
log(f"t={t.statistic:.2f}, df={t.df:.0f}, p={t.pvalue:.2e}, Cohen d={d:.2f}")

log("\n=== 3. POISSON-ТЕСТ (counts ~ Poisson) + NB как робастность ===")
# Дисперсия вычисляется единой функцией dispersion_ratio (задача 6),
# результат — один источник истины для всех отчётов.
vm_free = dispersion_ratio(free)
vm_ultra = dispersion_ratio(ultra)
OVERDISP_THR = 1.5
overdisp = (vm_free > OVERDISP_THR) or (vm_ultra > OVERDISP_THR)
log(f"var/mean free={vm_free:.2f}, ultra={vm_ultra:.2f} "
    f"(порог сверхдисперсии {OVERDISP_THR:.1f}) -> "
    f"{'СВЕРХДИСПЕРСИЯ: условный Poisson-тест ориентировочный, опирайтесь на MWU/NB' if overdisp else 'эквидисперсия, условие Пуассона выполнено'}.")
# Точный тест равенства rates: при H0 total_free ~ Binomial(total, exp_free/total_exp),
# где экспозиция = число пользователей x 12 мес (наблюдение полное за 2021).
# F-3: допущение равной экспозиции — в users_go.csv нет даты регистрации/оттока,
# поэтому каждый пользователь считается наблюдавшимся весь период.
total_free, total_ultra = int(free.sum()), int(ultra.sum())
total = total_free + total_ultra
p_exp = len(free) / (len(free) + len(ultra))
bt = stats.binomtest(total_free, total, p_exp, alternative="two-sided")
rate_free, rate_ultra = free.mean(), ultra.mean()
log(f"Суммарно поездок: free={total_free}, ultra={total_ultra}, total={total}")
log(f"Ожидаемая доля free при H0 (по экспозиции): {p_exp:.3f}, "
    f"наблюдаемая: {total_free / total:.3f}")
log(f"Binomial exact (условный Poisson-тест): p={bt.pvalue:.2e}")
log(f"Rate ratio ultra/free = {rate_ultra / rate_free:.3f} "
    f"(rates: free={rate_free:.2f}, ultra={rate_ultra:.2f} поездок/польз/год)")
try:
    import statsmodels.api as sm  # noqa
    import statsmodels.formula.api as smf
    df = rpu.copy()
    df["is_ultra"] = (df.subscription_type == "ultra").astype(int)
    nb = smf.glm("n ~ is_ultra", data=df,
                 family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    irr = float(np.exp(nb.params["is_ultra"]))
    ci = np.exp(nb.conf_int().loc["is_ultra"].to_numpy())
    log(f"NB GLM (робастность): IRR ultra/free={irr:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
        f"p={nb.pvalues['is_ultra']:.2e}")
except ImportError:
    irr = ultra.mean() / free.mean()
    # Delta-метод для log(IRR): SE^2 = var1/(n1*m1^2) + var0/(n0*m0^2)
    se_log = np.sqrt(free.var(ddof=1) / (n1 * free.mean() ** 2)
                     + ultra.var(ddof=1) / (n2 * ultra.mean() ** 2))
    lo, hi = np.exp(np.log(irr) - 1.96 * se_log), np.exp(np.log(irr) + 1.96 * se_log)
    log("statsmodels отсутствует -> IRR как отношение средних, CI дельта-методом:")
    log(f"IRR ultra/free={irr:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

log("\n=== 4. КОНТРОЛЬ СЕЗОННОСТИ: помесячная стратификация ===")
rides2 = rides.merge(users[["user_id", "subscription_type"]], on="user_id")
rides2["month"] = rides2["date"].dt.to_period("M").astype(str)
nu = users.groupby("subscription_type").size()
pm = rides2.groupby(["month", "subscription_type"]).size().unstack().div(nu, axis=1)
pm["ratio_free_ultra"] = pm["free"] / pm["ultra"]
log(pm.round(3).to_string())
log("Отношение free/ultra по месяцам: " + ", ".join(f"{v:.2f}" for v in pm["ratio_free_ultra"]))
# Помесячный MWU на уровне человеко-месяцев (с нулями)
all_users = users[["user_id", "subscription_type"]]
months = sorted(rides2["month"].unique())
month_rows = []
month_pvals = []
for m in months:
    riders = rides2.loc[rides2.month == m].groupby("user_id").size().reset_index(name="nm")
    mm = all_users.merge(riders, on="user_id", how="left").fillna({"nm": 0})
    f = mm.loc[mm.subscription_type == "free", "nm"].to_numpy()
    u = mm.loc[mm.subscription_type == "ultra", "nm"].to_numpy()
    p = stats.mannwhitneyu(f, u, alternative="two-sided").pvalue
    month_rows.append((m, float(f.mean()), float(u.mean()), float(p)))
    month_pvals.append(float(p))
# F-4: Holm-поправка на 12 помесячных сравнений
month_adj = holm_adjust(month_pvals)
sig_months = sum(a < 0.05 for a in month_adj)
for (m, mf, mu, p), pa in zip(month_rows, month_adj):
    log(f"{m}: mean free={mf:.3f}, ultra={mu:.3f}, MWU p={p:.2e}, p_Holm={pa:.2e}")
log(f"Месяцев с p_Holm<0.05: {sig_months}/{len(months)} (эффект одного знака во все месяцы)")

log("\n=== 5. КОНТРОЛЬ ГОРОДА: стратифицированный MWU ===")
# F-4: Holm-поправка на сравнения по городам
city_rows, city_pvals = [], []
for c, g in rpu.groupby("city"):
    f = g.loc[g.subscription_type == "free", "n"].to_numpy()
    u = g.loc[g.subscription_type == "ultra", "n"].to_numpy()
    if len(f) > 5 and len(u) > 5:
        p = stats.mannwhitneyu(f, u, alternative="two-sided").pvalue
        city_rows.append((c, float(f.mean()), len(f), float(u.mean()), len(u), float(p)))
        city_pvals.append(float(p))
city_adj = holm_adjust(city_pvals) if city_pvals else []
for (c, mf, nf, mu, nu_, p), pa in zip(city_rows, city_adj):
    log(f"{c}: free mean={mf:.2f} (n={nf}), ultra mean={mu:.2f} (n={nu_}), "
        f"p={p:.2e}, p_Holm={pa:.2e}")

log("\n=== 6. ДАТА РЕЛИЗА ТАРИФОВ / DiD ===")
log("ВНИМАНИЕ: этот блок относится к релизу ТАРИФОВ (RELEASE_DATE), а не к "
    "обновлению серверов. Гипотеза про серверы и обращения в поддержку — "
    "в отдельном скрипте gofast_support_test.py (данных в выгрузке нет, блокер).")
if RELEASE_DATE is None:
    log("RELEASE_DATE не задана: тариф наблюдается весь период, сравнение "
        "кросс-секционное. Причинный вывод невозможен (self-selection в тариф).")
    log("Когда дата T0 известна: задайте RELEASE_DATE, скрипт разделит rides на "
        "pre/post, нормирует на человеко-месяцы экспозиции и посчитает "
        "Difference-in-Differences с month-дамми для сезонности.")
else:
    t0 = pd.to_datetime(RELEASE_DATE)
    rides2["period"] = np.where(rides2["date"] < t0, "pre", "post")
    # Экспозиция: человеко-месяцы наблюдения пользователя в периоде.
    # Дат регистрации/оттока нет -> каждый пользователь наблюдается весь год.
    months_all = sorted(rides2["month"].unique())
    n_pre_months = sum(m < str(t0)[:7] for m in months_all)
    n_post_months = len(months_all) - n_pre_months
    counts = rides2.groupby(["period", "subscription_type"]).size().unstack()
    n_users = users.groupby("subscription_type").size()
    log(counts.to_string())
    # rate = поездок / (пользователи × месяцы периода)
    rate = counts.copy().astype(float)
    rate.loc["pre"] = rate.loc["pre"] / (n_users * max(n_pre_months, 1))
    rate.loc["post"] = rate.loc["post"] / (n_users * max(n_post_months, 1))
    log("\nRate на человеко-месяц:")
    log(rate.round(3).to_string())
    # DiD = (ultra_post - ultra_pre) - (free_post - free_pre)
    did = ((rate.loc["post", "ultra"] - rate.loc["pre", "ultra"])
           - (rate.loc["post", "free"] - rate.loc["pre", "free"]))
    log(f"\nDiD (ultra - free) на человеко-месяц: {did:+.4f} поездок")
    log(f"Месяцев pre/post: {n_pre_months}/{n_post_months}. "
        "Для корректного причинного вывода нужны проверка предтрендов и дата T0; "
        "без них DiD — описательная декомпозиция.")

(OUT / "counts_test.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nОтчет сохранен: {OUT / 'counts_test.txt'}")
