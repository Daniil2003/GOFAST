# -*- coding: utf-8 -*-
"""Общие статистические утилиты GoFast: единый источник истины для var/mean и поправок.

Используется всеми консольными скриптами, чтобы числа не расходились
между отчётами (задача 6 постановки).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
DS = BASE / "datasets"
OUT = BASE / "output"


def dispersion_ratio(values) -> float:
    """Отношение выборочной дисперсии к среднему (индекс сверхдисперсии).

    Для counts: значение ≈ 1 — пуассоновский режим, >> 1 — сверхдисперсия.
    Единая реализация для всех скриптов проекта.
    """
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    return float(arr.var(ddof=1) / mean)


def overdispersed(values, threshold: float = 1.5) -> bool:
    """True, если индекс сверхдисперсии превышает порог."""
    return dispersion_ratio(values) > threshold


def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down коррекция p-value без внешних зависимостей."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        cur = (m - rank) * float(p[idx])
        cur = max(cur, prev)
        prev = cur
        adj[idx] = min(cur, 1.0)
    return adj


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает users / rides / subs, чистит дубли user_id."""
    users = pd.read_csv(DS / "users_go.csv").drop_duplicates("user_id")
    rides = pd.read_csv(DS / "rides_go.csv", parse_dates=["date"])
    subs = pd.read_csv(DS / "subscriptions_go.csv")
    return users, rides, subs


def price_map(subs: pd.DataFrame) -> dict:
    """subscription_type -> (start_ride_price, minute_price, subscription_fee)."""
    return {
        row["subscription_type"]: (
            float(row["start_ride_price"]),
            float(row["minute_price"]),
            float(row["subscription_fee"]),
        )
        for _, row in subs.iterrows()
    }


def ride_revenue_vectorized(df: pd.DataFrame, prices: dict) -> np.ndarray:
    """Векторизованный расчёт выручки поездки: start + minute × duration."""
    start = df["subscription_type"].map(lambda t: prices[t][0]).to_numpy(dtype=float)
    rate = df["subscription_type"].map(lambda t: prices[t][1]).to_numpy(dtype=float)
    return start + rate * df["duration"].to_numpy(dtype=float)
