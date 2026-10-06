# -*- coding: utf-8 -*-
"""GoFast — гипотеза «обновление серверов снизило обращения в техподдержку».

Запуск: python3 gofast_support_test.py
Требуется: pandas, numpy, scipy.

СТАТУС: ДАННЫХ НЕТ -> зафиксирован блокер. Скрипт НЕ фабрикует результат.

Что нужно из данных (в текущей выгрузке отсутствует):
  1. Таблица обращений в поддержку: как минимум (дата/период, число обращений).
     Ожидаемый путь: datasets/support_go.csv
  2. Дата деплоя обновления серверов T0.
  3. Желательно контрольная группа: обращения по темам, НЕ связанным с серверами
     (для Difference-in-Differences).

Методы, подготовленные на будущее (реализуются, когда появятся данные):
  * DiD с контролем (тема-контроль) и проверкой ПРЕДТРЕНДОВ;
  * без контроля — interrupted time series (ITS) / change-point, а НЕ просто
    «до/после» (иначе сезонность и общий тренд дадут ложное «снижение»).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DS = BASE / "datasets"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

SUPPORT_FILE = DS / "support_go.csv"
T0_DEPLOY = None  # дата обновления серверов 'YYYY-MM-DD', когда станет известна

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def describe_blocker():
    log("=" * 72)
    log("GoFast — ОБРАЩЕНИЯ В ПОДДЕРЖКУ: анализ невозможен (нет данных)")
    log("=" * 72)
    log("")
    log("Проверка гипотезы «обновление серверов снизило обращения в техподдержку» "
        "НЕ выполнена, потому что в выгрузке нет необходимых данных:")
    log(f"  1) нет таблицы обращений (ожидается файл: "
        f"{SUPPORT_FILE.relative_to(BASE)});")
    log("  2) нет даты обновления серверов T0;")
    log("  3) нет контрольной группы (темы обращений, не связанные с серверами).")
    log("")
    log("Что предоставить для будущего анализа:")
    log("  - datasets/support_go.csv с колонками:")
    log("      date        (дата или период, YYYY-MM-DD)")
    log("      tickets     (число обращений) — если агрегировано по дням/месяцам;")
    log("      тема/категория  (для DiD-контроля: серверы vs прочие темы);")
    log("  - дату деплоя T0 (обновление серверов).")
    log("")
    log("Готовые методы (включаются автоматически при наличии данных):")
    log("  * Есть контроль -> Difference-in-Differences (тема×период), "
        "с проверкой предтрендов до T0.")
    log("  * Нет контроля -> interrupted time series (ITS)/change-point "
        "с выделением тренда и сезонности; вывод «до/после» без этого некорректен.")
    log("")
    log("ВЫВОД: результат честно помечен как «нет данных». Гипотеза НЕ "
        "опровергнута и НЕ подтверждена — она не проверяема на текущей выгрузке.")
    log("")


def run_did(tickets: pd.DataFrame, t0: str):
    """DiD с контрольной группой: тема×период."""
    t0d = pd.to_datetime(t0)
    tickets = tickets.copy()
    tickets["date"] = pd.to_datetime(tickets["date"])
    tickets["period"] = np.where(tickets["date"] < t0d, "pre", "post")
    pivot = tickets.pivot_table(index="period", columns="topic",
                                values="tickets", aggfunc="sum")
    log("Таблица период×тема (обращения):")
    log(pivot.to_string())
    if {"server", "other"}.issubset(pivot.columns) and {"pre", "post"}.issubset(pivot.index):
        did = ((pivot.loc["post", "server"] - pivot.loc["pre", "server"])
               - (pivot.loc["post", "other"] - pivot.loc["pre", "other"]))
        log(f"\nDiD (server - other) = {did:+.1f} обращений")
        log("Проверка предтрендов (до T0) обязательна; при наличии — добавить "
            "в модель месяц/сезонность.")
    else:
        log("Для DiD нужны темы 'server' и 'other' и обе фазы pre/post.")


def run_its(tickets: pd.DataFrame, t0: str):
    """ITS/change-point без контроля: тренд + сдвиг после T0."""
    ts = tickets.sort_values("date").copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts["t"] = np.arange(len(ts))
    ts["post_t0"] = (ts["date"] >= pd.to_datetime(t0)).astype(int)
    # Простая OLS: tickets ~ t + post_t0  (при statsmodels — с HAC-ошибками)
    X = np.column_stack([np.ones(len(ts)), ts["t"], ts["post_t0"]])
    y = ts["tickets"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    log("ITS (tickets ~ t + post_t0):")
    log(f"  базовый уровень={beta[0]:.2f}, тренд/период={beta[1]:+.3f}, "
        f"сдвиг после T0={beta[2]:+.2f}")
    log("Оговорка: без контроля и сезонности сдвиг может отражать общий тренд, "
        "а не эффект обновления серверов. Рекомендуется change-point/ARIMA.")


def main():
    if not SUPPORT_FILE.exists():
        describe_blocker()
        (OUT / "support_test.txt").write_text("\n".join(lines), encoding="utf-8")
        log(f"Отчёт сохранён: {OUT / 'support_test.txt'}")
        # Явный блокер: не тихий пустой результат, а понятная ошибка (код возврата 0,
        # т.к. блокер ожидаем и зафиксирован; при желании поднять до sys.exit(2)).
        log("СТАТУС: BLOCKED (ожидается datasets/support_go.csv и дата T0).")
        return 0

    tickets = pd.read_csv(SUPPORT_FILE)
    required = {"date", "tickets", "topic"}
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(
            f"Файл {SUPPORT_FILE.name} найден, но не хватает колонок: {sorted(missing)}. "
            f"Ожидаются: {sorted(required)}."
        )

    log("=" * 72)
    log("GoFast — ОБРАЩЕНИЯ В ПОДДЕРЖКУ: анализ")
    log("=" * 72)
    if T0_DEPLOY is None:
        raise ValueError(
            "Файл обращений найден, но не задана дата обновления серверов T0. "
            "Установите T0_DEPLOY = 'YYYY-MM-DD'."
        )
    run_did(tickets, T0_DEPLOY)
    log("")
    run_its(tickets, T0_DEPLOY)
    (OUT / "support_test.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"Отчёт сохранён: {OUT / 'support_test.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
