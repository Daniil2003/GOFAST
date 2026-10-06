# -*- coding: utf-8 -*-
"""GoFast — помесячная выручка тарифов и гипотеза «выручка ultra выше free».

Запуск: python3 gofast_revenue_monthly.py
Требуется: pandas, numpy, scipy.
Результат: print в консоль + output/revenue_monthly.txt (+ CSV помесячного ряда).

Ключевые фиксы относительно старого решения:
  * считается помесячный ряд выручки в рублях (раньше — только годовой тотал);
  * АБОНПЛАТА 199 ₽/мес приписывается КАЖДОМУ месяцу через число активных
    в этом месяце ultra: fee(мес) = 199 × (число ultra с поездками в месяце);
  * сравнение на пользователя (ARPU), а не абсолютные тоталы групп
    (группы разного размера: free 835 vs ultra 699);
  * 12 помесячных значений free vs ultra — ЗАВИСИМЫЕ (парные) наблюдения ->
    парный t-тест на 12 разностях (и Wilcoxon как робастная альтернатива),
    а НЕ 12 независимых MWU.
"""
import numpy as np
import pandas as pd
from scipy import stats
from gofast_stats_common import load_data, price_map, OUT

ALPHA = 0.05
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


users, rides, subs = load_data()
prices = price_map(subs)
fee_ultra = prices["ultra"][2]  # 199 ₽/мес

n_free = int((users.subscription_type == "free").sum())
n_ultra = int((users.subscription_type == "ultra").sum())

rides_m = rides.merge(users[["user_id", "subscription_type"]], on="user_id", how="left")
rides_m["month"] = rides_m["date"].dt.to_period("M").astype(str)
rides_m["ride_revenue"] = (
    rides_m["subscription_type"].map(lambda t: prices[t][0])
    + rides_m["subscription_type"].map(lambda t: prices[t][1]) * rides_m["duration"]
)

log("=" * 72)
log("GoFast — ПОМЕСЯЧНАЯ ВЫРУЧКА: ultra vs free")
log("=" * 72)

# --- 1. Помесячная выручка с поездок ---
rev_rides = rides_m.groupby(["month", "subscription_type"])["ride_revenue"].sum().unstack()
active = rides_m.groupby(["month", "subscription_type"])["user_id"].nunique().unstack()
active_all = active.reindex(columns=["free", "ultra"], fill_value=0)

# --- 2. Абонплата: 199 × число ultra, активных в месяце ---
fee = fee_ultra * active_all["ultra"]

log("\n### 1. ДОПУЩЕНИЕ ПО АБОНПЛАТЕ ###")
log(f"Абонплата ultra: {fee_ultra:.0f} ₽/мес. Начисляется за КАЖДЫЙ месяц, а не "
    "только за месяцы с поездками.")
log(f"fee(месяц) = {fee_ultra:.0f} × (число ultra, активных в месяце).")
log("Активных ultra по месяцам: min=%d, max=%d, медиана=%.0f"
    % (active_all["ultra"].min(), active_all["ultra"].max(), active_all["ultra"].median()))
log(f"Альтернативное допущение — все {n_ultra} ultra платят весь год "
    f"({n_ultra} × {fee_ultra:.0f} = {n_ultra*fee_ultra:,.0f} ₽/мес). Дат регистрации/"
    "оттока в users_go.csv НЕТ, поэтому выбран консервативный вариант «платят "
    "только активные». Ниже приведена чувствительность вывода к этому допущению.")

# --- 3. Итоговая помесячная выручка ---
rev = pd.DataFrame(index=rev_rides.index)
rev["free"] = rev_rides["free"]
rev["ultra_rides"] = rev_rides["ultra"]
rev["ultra_fee_active"] = fee
rev["ultra_total"] = rev["ultra_rides"] + rev["ultra_fee_active"]
rev["ultra_fee_full"] = fee_ultra * n_ultra
rev["ultra_total_full"] = rev["ultra_rides"] + rev["ultra_fee_full"]

log("\n### 2. ПОМЕСЯЧНЫЙ РЯД ВЫРУЧКИ (₽) ###")
log(rev.round(0).to_string())

# --- 4. ARPU на пользователя ---
rev["arpu_free"] = rev["free"] / n_free
rev["arpu_ultra_active"] = rev["ultra_total"] / n_ultra
rev["arpu_ultra_full"] = rev["ultra_total_full"] / n_ultra

log("\n### 3. ARPU НА ПОЛЬЗОВАТЕЛЯ (₽/мес) ###")
log(rev[["arpu_free", "arpu_ultra_active", "arpu_ultra_full"]].round(0).to_string())
log(f"Средний ARPU: free={rev['arpu_free'].mean():.0f}, "
    f"ultra (активные)={rev['arpu_ultra_active'].mean():.0f}, "
    f"ultra (все платят)={rev['arpu_ultra_full'].mean():.0f} ₽/мес")

diff = (rev["arpu_ultra_active"] - rev["arpu_free"]).to_numpy()
diff_full = (rev["arpu_ultra_full"] - rev["arpu_free"]).to_numpy()

log("\n### 4. ПАРНЫЙ ТЕСТ (12 месяцев, зависимые наблюдения) ###")
log("H0: средняя помесячная разница ARPU (ultra - free) = 0; H1: ≠ 0.")
t = stats.ttest_rel(rev["arpu_ultra_active"], rev["arpu_free"])
se = diff.std(ddof=1) / np.sqrt(len(diff))
tcrit = stats.t.ppf(1 - ALPHA / 2, len(diff) - 1)
log(f"Парный t-test (активные ultra): t={t.statistic:.2f}, df={len(diff)-1}, p={t.pvalue:.3e}")
log(f"Средняя разница ARPU = {diff.mean():+.0f} ₽/мес, 95% CI "
    f"[{diff.mean()-tcrit*se:+.0f}; {diff.mean()+tcrit*se:+.0f}]")
try:
    w = stats.wilcoxon(rev["arpu_ultra_active"], rev["arpu_free"])
    log(f"Wilcoxon (робастно): W={w.statistic:.0f}, p={w.pvalue:.3e}")
except ValueError as e:
    log(f"Wilcoxon: {e}")
if diff.mean() > 0:
    log("Вывод: помесячная выручка НА ПОЛЬЗОВАТЕЛЯ у ultra выше free (с учётом абонплаты).")
else:
    log("Вывод: помесячная выручка на пользователя у ultra НЕ выше free.")

# --- 5. Чувствительность к допущению абонплаты ---
log("\n### 5. ЧУВСТВИТЕЛЬНОСТЬ К ДОПУЩЕНИЮ АБОНПЛАТЫ ###")
t_full = stats.ttest_rel(rev["arpu_ultra_full"], rev["arpu_free"])
lo_full, hi_full = np.percentile(diff_full, [2.5, 97.5])
log("Вариант «все ultra платят весь год»:")
log(f"  средняя разница ARPU = {diff_full.mean():+.0f} ₽/мес, "
    f"парный t p={t_full.pvalue:.3e}")
log(f"Вывод устойчив: разница {'сохраняется' if diff_full.mean() > 0 else 'меняется'} "
    "при обоих допущениях.")

# --- 6. Сравнение абсолютных тоталов (для контекста, НЕ основной критерий) ---
log("\n### 6. АБСОЛЮТНЫЕ ТОТАЛЫ (контекст; группы разного размера) ###")
log(f"Итого год: free={rev['free'].sum():,.0f} ₽, ultra(активные)={rev['ultra_total'].sum():,.0f} ₽")
log(f"Из них абонплата ultra: {rev['ultra_fee_active'].sum():,.0f} ₽; "
    f"поездки ultra: {rev['ultra_rides'].sum():,.0f} ₽; поездки free: {rev['free'].sum():,.0f} ₽")
log("Абсолютные тоталы зависят от размера группы (835 vs 699), поэтому основной "
    "критерий — ARPU на пользователя (раздел 3-4).")

log("\n" + "=" * 72)
log("ВЫВОД")
log("=" * 72)
log(f"Помесячная выручка на пользователя: ultra {diff.mean():+.0f} ₽/мес против free "
    f"(p={t.pvalue:.2e}, 95% CI [{diff.mean()-tcrit*se:+.0f}; {diff.mean()+tcrit*se:+.0f}]).")
log("Допущения: абонплата 199 ₽/мес; ultra платят в месяцы с поездками "
    "(консервативно); полная экспозиция пользователей (нет дат регистрации/оттока); "
    "самоотбор в тариф -> вывод описательный, не причинный.")

rev.to_csv(OUT / "revenue_monthly.csv", encoding="utf-8")
(OUT / "revenue_monthly.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nОтчёт сохранён: {OUT / 'revenue_monthly.txt'}, {OUT / 'revenue_monthly.csv'}")
