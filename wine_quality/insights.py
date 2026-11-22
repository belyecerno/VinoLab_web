"""Расчёты для интерфейса: пресеты, вклад показателей в прогноз, процентили, диапазоны ползунков."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, MODEL_FEATURES, TARGET


def slider_specs(df: pd.DataFrame) -> dict:
    """Диапазон (0.5–99.5 перцентиль), шаг и формат для каждого показателя."""
    specs = {}
    for f in FEATURES:
        lo, hi = df[f].quantile([0.005, 0.995])
        if f == "density":
            step, fmt, nd = 0.0001, "%.4f", 4
        elif f in ("chlorides",):
            step, fmt, nd = 0.001, "%.3f", 3
        elif hi - lo > 50:
            step, fmt, nd = 1.0, "%.0f", 0
        elif hi - lo > 5:
            step, fmt, nd = 0.1, "%.1f", 1
        else:
            step, fmt, nd = 0.01, "%.2f", 2
        specs[f] = {"min": round(float(lo), nd), "max": round(float(hi), nd),
                    "step": step, "format": fmt, "nd": nd}
    return specs


def presets(df: pd.DataFrame) -> dict:
    """Готовые профили проб: медианы по типу и по уровню качества."""
    out = {}
    for t, ru in (("white", "белое"), ("red", "красное")):
        s = df[df["type"] == t]
        out[f"Типичное {ru}"] = {"type": t, **s[FEATURES].median().to_dict()}
    for t, ru in (("white", "белое"), ("red", "красное")):
        s = df[(df["type"] == t) & (df[TARGET] >= 7)]
        out[f"Эталон {ru} (≥7)"] = {"type": t, **s[FEATURES].median().to_dict()}
    s = df[df[TARGET] <= 4]
    t = s["type"].mode()[0]
    out["Проблемная проба (≤4)"] = {"type": t, **s[s["type"] == t][FEATURES].median().to_dict()}
    return out


def reference_profile(df: pd.DataFrame, wine_type: str) -> pd.Series:
    s = df[(df["type"] == wine_type) & (df[TARGET] >= 7)]
    return s[FEATURES].median()


def contributions(model, x: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Насколько изменится прогноз, если заменить показатель типичным значением для этого типа вина.
    Положительное значение — показатель поднимает оценку относительно типичной пробы."""
    wine_type = "red" if int(x["is_red"]) == 1 else "white"
    med = df[df["type"] == wine_type][FEATURES].median()
    base = pd.DataFrame([x[MODEL_FEATURES]] * (len(FEATURES) + 1)).reset_index(drop=True)
    for i, f in enumerate(FEATURES, start=1):
        base.loc[i, f] = med[f]
    preds = model.predict(base[MODEL_FEATURES])
    contrib = preds[0] - preds[1:]
    out = pd.DataFrame({"feature": FEATURES, "value": [x[f] for f in FEATURES],
                        "typical": med.values, "effect": contrib})
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def percentiles(x: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Процентиль каждого показателя пробы среди вин того же типа (0–100)."""
    wine_type = "red" if int(x["is_red"]) == 1 else "white"
    s = df[df["type"] == wine_type]
    return pd.Series({f: float((s[f] < x[f]).mean() * 100 + (s[f] == x[f]).mean() * 50) for f in FEATURES})


def dataset_predictions(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[MODEL_FEATURES])


def rank_among(pred: float, all_preds: np.ndarray) -> float:
    """Доля образцов датасета, у которых прогноз ниже, %."""
    return float((all_preds < pred).mean() * 100)


def triage_curve(test: pd.DataFrame, boundary: float = 5.5) -> pd.DataFrame:
    """Экономика предварительного отбора по тестовой выборке.
    Проба считается «уверенной», если прогноз отстоит от границы «брак (≤5) / норма (≥6)» не меньше чем на d.
    Для каждого d: доля уверенных проб и точность их разметки (на данных, которых модель не видела)."""
    y, p = test["y_true"].values, test["y_pred"].values
    rows = []
    for d in np.round(np.arange(0.0, 1.51, 0.05), 2):
        conf = np.abs(p - boundary) >= d
        if conf.sum() < 20:
            break
        acc = float(((p[conf] < boundary) == (y[conf] <= 5)).mean())
        rows.append({"d": d, "share": float(conf.mean()), "accuracy": acc})
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target_accuracy: float) -> pd.Series:
    """Наибольшая доля уверенных проб при точности не ниже заданной."""
    ok = curve[curve["accuracy"] >= target_accuracy]
    return (ok.iloc[0] if len(ok) else curve.iloc[-1])
