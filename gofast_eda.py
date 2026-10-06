"""
GoFast — исследовательский анализ (EDA)
Запуск: python3 gofast_eda.py
Требуется: pandas, matplotlib, seaborn, numpy
Результат: папка output/ с 10 графиками (PNG) + print метрик в консоль
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent
DS = BASE / "datasets"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130

users = pd.read_csv(DS / "users_go.csv")
rides = pd.read_csv(DS / "rides_go.csv")
subs = pd.read_csv(DS / "subscriptions_go.csv")

print("=== РАЗМЕРЫ ===")
print(users.shape, rides.shape, subs.shape)
print(subs.to_string(index=False))

# --- Качество данных ---
print("\n=== ДУБЛИ user_id ===")
dups = users[users.duplicated("user_id", keep=False)].sort_values("user_id")
print(f"Дублей строк: {len(dups)}, уникальных user_id: {users['user_id'].nunique()}")
if len(dups):
    print(dups.head(10).to_string(index=False))

print("\n=== Пропуски ===")
print(users.isna().sum())
print(rides.isna().sum())

print("\n=== Диапазоны ===")
print("Возраст:", users.age.min(), users.age.max(), users.age.median())
print("Даты поездок:", rides.date.min(), rides.date.max())

# --- Контроль качества (робастный, без мифических констант) ---
print("\n=== QC: квантили duration / distance ===")
for col in ["duration", "distance"]:
    q = rides[col].quantile([0.001, 0.01, 0.99, 0.999])
    print(f"{col}: min={rides[col].min():.3f}, 0.1%={q[0.001]:.3f}, 1%={q[0.01]:.3f}, "
          f"медиана={rides[col].median():.3f}, 99%={q[0.99]:.3f}, 99.9%={q[0.999]:.3f}, "
          f"max={rides[col].max():.3f}")

print("\n=== QC: физически неразумные скорости ===")
# speed в м/мин -> км/ч: *60/1000
speed_kmh = rides["distance"] / rides["duration"] * 60 / 1000
SPEED_MIN_KMH, SPEED_MAX_KMH = 2.0, 25.0
too_slow = (speed_kmh < SPEED_MIN_KMH)
too_fast = (speed_kmh > SPEED_MAX_KMH)
print(f"speed (км/ч): min={speed_kmh.min():.2f}, медиана={speed_kmh.median():.2f}, max={speed_kmh.max():.2f}")
print(f"Скорость < {SPEED_MIN_KMH} км/ч: {too_slow.sum()} ({too_slow.mean():.4%})")
print(f"Скорость > {SPEED_MAX_KMH} км/ч: {too_fast.sum()} ({too_fast.mean():.4%})")
print(f"Доля поездок duration < 1 мин: {(rides.duration < 1).sum()} ({(rides.duration < 1).mean():.4%})")

print("\n=== QC: вырожденные значения и дубликаты ===")
print(f"distance <= 0: {(rides.distance <= 0).sum()}")
print(f"duration <= 0: {(rides.duration <= 0).sum()}")
print(f"Полные дубликаты строк поездок: {rides.duplicated().sum()}")
print(f"Дубликаты (user_id, date, distance, duration): "
      f"{rides.duplicated(['user_id', 'date', 'distance', 'duration']).sum()}")

rides["date"] = pd.to_datetime(rides["date"])
users_clean = users.drop_duplicates("user_id", keep="first")

# --- Тарифная экономика ---
price = subs.set_index("subscription_type").to_dict("index")
# revenue за поездку
def ride_revenue(row_user_type, duration):
    p = price[row_user_type]
    return p["start_ride_price"] + p["minute_price"] * duration

rides_m = rides.merge(users_clean[["user_id", "subscription_type", "city", "age"]],
                      on="user_id", how="left")
rides_m["ride_revenue"] = rides_m.apply(
    lambda r: ride_revenue(r["subscription_type"], r["duration"]), axis=1)
rides_m["speed_m_min"] = rides_m["distance"] / rides_m["duration"]
rides_m["month"] = rides_m["date"].dt.to_period("M").astype(str)
rides_m["weekday"] = rides_m["date"].dt.day_name()

# Break-even ultra vs free: n = 199 / (50 + 2*m)
for m in [10, 15, 17, 20, 25]:
    print(f"Break-even при средней длительности {m} мин: {199/(50+2*m):.2f} поездок/мес")

# --- Агрегаты ---
print("\n=== Пользователи по тарифу ===")
print(users_clean.subscription_type.value_counts())
print(users_clean.subscription_type.value_counts(normalize=True))

print("\n=== По городам ===")
print(users_clean.city.value_counts())

print("\n=== Поездки на пользователя ===")
rpu = rides_m.groupby("user_id").size()
print(rpu.describe())
print("По тарифу (среднее поездок/польз):")
print(rides_m.groupby("subscription_type")["user_id"].count() /
      users_clean.groupby("subscription_type")["user_id"].count())

print("\n=== Выручка ===")
rev_rides = rides_m.groupby("subscription_type")["ride_revenue"].sum()
print(rev_rides)
# subscription fee: гипотеза — 199 руб/мес => 2388/год на ultra-пользователя
ULTRA_FEE_YEAR = 199 * 12
ultra_n = (users_clean.subscription_type == "ultra").sum()
print(f"Абонплата ultra (199*12*{ultra_n}): {ULTRA_FEE_YEAR*ultra_n:,.0f}")
print(f"Итого ultra (поездки+абонплата): {rev_rides.get('ultra',0)+ULTRA_FEE_YEAR*ultra_n:,.0f}")
print(f"Итого free (только поездки): {rev_rides.get('free',0):,.0f}")
print(f"ARPU год (с абонплатой): ultra={(rev_rides.get('ultra',0)+ULTRA_FEE_YEAR*ultra_n)/max(ultra_n,1):,.0f}, "
      f"free={rev_rides.get('free',0)/max((users_clean.subscription_type=='free').sum(),1):,.0f}")

# ================= ГРАФИКИ =================
# 1. Распределение пользователей по тарифу и городам
fig, ax = plt.subplots(figsize=(8, 4.5))
pd.crosstab(users_clean.city, users_clean.subscription_type).plot(kind="bar", ax=ax)
ax.set_title("Пользователи по городам и тарифам")
ax.set_ylabel("Пользователей")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(OUT / "01_users_city_tariff.png"); plt.close()

# 2. Возрастное распределение
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(users_clean, x="age", hue="subscription_type", bins=25, kde=True, ax=ax)
ax.set_title("Возраст пользователей по тарифам")
plt.tight_layout(); plt.savefig(OUT / "02_age_dist.png"); plt.close()

# 3. Несовершеннолетние по городам (риск)
minor = users_clean.assign(is_minor=users_clean.age < 18)
fig, ax = plt.subplots(figsize=(8, 4.5))
minor.groupby("city")["is_minor"].mean().sort_values().plot(kind="barh", ax=ax)
ax.set_title("Доля несовершеннолетних (<18) по городам")
ax.set_xlabel("Доля"); plt.tight_layout()
plt.savefig(OUT / "03_minors_city.png"); plt.close()

# 4. Динамика поездок по месяцам + тариф
fig, ax = plt.subplots(figsize=(9, 4.5))
rides_m.groupby(["month", "subscription_type"]).size().unstack().plot(ax=ax, marker="o")
ax.set_title("Поездки по месяцам (сезонность)")
ax.set_ylabel("Поездок"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(OUT / "04_monthly_rides.png"); plt.close()

# 5. Распределение длительности и дистанции
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
sns.histplot(rides_m[rides_m.duration < 40], x="duration", hue="subscription_type", bins=30, ax=axes[0], kde=True)
axes[0].set_title("Длительность поездки (мин, <40)")
sns.histplot(rides_m[rides_m.distance < 7000], x="distance", hue="subscription_type", bins=30, ax=axes[1], kde=True)
axes[1].set_title("Дистанция (м, <7000)")
plt.tight_layout(); plt.savefig(OUT / "05_duration_distance.png"); plt.close()

# 6. Поездок на пользователя (long-tail)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(rpu, bins=30, kde=True, ax=ax)
ax.axvline(rpu.mean(), color="red", linestyle="--", label=f"mean={rpu.mean():.1f}")
ax.set_title("Распределение: поездок на пользователя за 2021")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "06_rides_per_user.png"); plt.close()

# 7. Выручка с поездки
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(rides_m, x="ride_revenue", hue="subscription_type", bins=30, ax=ax)
ax.set_title("Выручка с одной поездки (без абонплаты)")
plt.tight_layout(); plt.savefig(OUT / "07_revenue_per_ride.png"); plt.close()

# 8. Средний чек и частота по городам
city_stats = rides_m.groupby("city").agg(rides=("user_id", "count"),
    avg_dur=("duration", "mean"), avg_rev=("ride_revenue", "mean"))
city_stats["users"] = users_clean.groupby("city").size()
city_stats["rpu"] = city_stats["rides"] / city_stats["users"]
fig, ax = plt.subplots(figsize=(8, 4.5))
city_stats.sort_values("rpu").plot(kind="barh", y="rpu", ax=ax, legend=False)
ax.set_title("Средняя частота (поездок/польз за год) по городам")
plt.tight_layout(); plt.savefig(OUT / "08_city_rpu.png"); plt.close()

# 9. Скорость (контроль качества)
fig, ax = plt.subplots(figsize=(8, 4.5))
sc = rides_m.sample(min(2000, len(rides_m)))
sns.scatterplot(data=sc, x="duration", y="distance", hue="subscription_type", alpha=0.5, ax=ax)
ax.set_title("Дистанция vs длительность")
ax.set_xlim(0, 40); plt.tight_layout()
plt.savefig(OUT / "09_distance_duration_scatter.png"); plt.close()

# 10. Возраст vs активность
act = rides_m.groupby("user_id").agg(n=("date", "count"), avg_dur=("duration", "mean"))
act = act.merge(users_clean[["user_id", "age", "subscription_type"]], on="user_id")
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(data=act.sample(min(1000, len(act))), x="age", y="n",
                hue="subscription_type", alpha=0.6, ax=ax)
ax.set_title("Возраст vs число поездок пользователя")
plt.tight_layout(); plt.savefig(OUT / "10_age_activity.png"); plt.close()

print(f"\nГрафики сохранены в {OUT}")
print(city_stats.round(2).to_string())
