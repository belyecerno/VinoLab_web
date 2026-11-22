"""Обучение, сравнение и сохранение моделей прогноза оценки качества вина.

Ключевые решения:
* дубликаты проб держим в одной части выборки (группированное разбиение) — иначе утечка;
* модель выбирается по R² на кросс-валидации, тестовая выборка используется только для отчёта;
* для каждого прогноза строится 90 % интервал (конформное предсказание с адаптивной шириной);
* для метрик на тесте считаются 95 % доверительные интервалы бутстрепом.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from .data import FEATURES, MODEL_FEATURES, TARGET

RANDOM_STATE = 42
Q_MIN, Q_MAX = 3, 9          # диапазон оценок, встречающихся в датасете
INTERVAL_LEVEL = 0.90        # уровень интервала прогноза
STANDARD_CV_MODELS = ("Extra Trees", "Случайный лес", "Градиентный бустинг", "Ансамбль (ET + бустинг + SVR)")


OPTIONAL_ERRORS: dict = {}  # почему необязательная библиотека не подключилась


def _log_scaled(est):
    """log1p + стандартизация: для моделей, чувствительных к масштабу и скошенным распределениям."""
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), est)


def candidate_models(include_optional: bool = True) -> dict:
    rs = RANDOM_STATE
    models = {
        "Базовая (среднее)": DummyRegressor(strategy="mean"),
        "Линейная регрессия (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "k ближайших соседей": _log_scaled(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "Нейросеть (MLP)": _log_scaled(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=800,
                                                    early_stopping=True, random_state=rs)),
        "Опорные векторы (SVR)": _log_scaled(SVR(C=3, gamma="scale", epsilon=0.1)),
        "Градиентный бустинг": HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=10,
            l2_regularization=1.0, random_state=rs),
        "Случайный лес": RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=rs),
        "Extra Trees": ExtraTreesRegressor(n_estimators=600, n_jobs=-1, random_state=rs),
        # усреднение разнотипных моделей: деревья + бустинг + SVR ошибаются по-разному.
        # Стекинг не используем: его внутренняя CV не учитывает дубликаты и переобучает мета-модель.
        "Ансамбль (ET + бустинг + SVR)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=600, n_jobs=-1, random_state=rs)),
            ("hgb", HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
                                                  min_samples_leaf=10, l2_regularization=1.0, random_state=rs)),
            ("svr", _log_scaled(SVR(C=3, gamma="scale", epsilon=0.1)))]),
    }
    if include_optional:  # популярные бустинги — если библиотеки установлены
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, subsample=0.8,
                                             colsample_bytree=0.8, n_jobs=-1, random_state=rs)
        except Exception as e:
            OPTIONAL_ERRORS["XGBoost"] = f"{type(e).__name__}: {e}"
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                               random_state=rs, verbose=-1)
        except Exception as e:
            OPTIONAL_ERRORS["LightGBM"] = f"{type(e).__name__}: {e}"
        try:
            from catboost import CatBoostRegressor
            models["CatBoost"] = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                                                   random_seed=rs, verbose=0)
        except Exception as e:
            OPTIONAL_ERRORS["CatBoost"] = f"{type(e).__name__}: {e}"
    return models


def round_score(x):
    """Обычное округление (0.5 вверх) с ограничением диапазоном шкалы."""
    return np.clip(np.floor(np.asarray(x, dtype=float) + 0.5), Q_MIN, Q_MAX).astype(int)


def quality_band(q: float) -> str:
    q = int(round_score(q))
    if q <= 5:
        return "низкое"
    if q == 6:
        return "среднее"
    return "высокое"


def regression_metrics(y_true, y_pred) -> dict:
    y_cls = round_score(y_pred)
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Точность (точное совпадение)": float(np.mean(y_cls == y_true)),
        "Точность (±1 балл)": float(np.mean(np.abs(y_cls - y_true) <= 1)),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> dict:
    """95 % доверительные интервалы R² и MAE на тесте (бутстреп по пробам)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2s, maes = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        r2s.append(r2_score(y_true[i], y_pred[i]))
        maes.append(mean_absolute_error(y_true[i], y_pred[i]))
    return {"R² ДИ низ": np.percentile(r2s, 2.5), "R² ДИ верх": np.percentile(r2s, 97.5),
            "MAE ДИ низ": np.percentile(maes, 2.5), "MAE ДИ верх": np.percentile(maes, 97.5)}


@dataclass
class TrainResult:
    model: object
    model_name: str
    metrics: pd.DataFrame            # сравнение всех моделей
    X_test: pd.DataFrame
    y_test: pd.Series
    test_type: pd.Series
    y_pred: np.ndarray
    feature_ranges: dict = field(default_factory=dict)
    interval: dict = field(default_factory=dict)


def _sigma_model():
    # модель «ожидаемой ошибки»: где проба нетипичная, интервал шире
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)


def fit_interval(X_tr, y_tr, oof_pred, groups, cv, X_te, y_te, pred_te, level=INTERVAL_LEVEL) -> dict:
    """Адаптивное конформное предсказание.
    1) |ошибка| вне выборки (out-of-fold) лучшей модели;
    2) отдельная модель предсказывает ожидаемую |ошибку| σ(x);
    3) q — квантиль нормированных ошибок |e|/σ; интервал = прогноз ± q·σ(x).
    Покрытие проверяется на отложенной выборке."""
    abs_res = np.abs(y_tr.values - oof_pred)
    sig_oof = np.clip(cross_val_predict(_sigma_model(), X_tr, abs_res, cv=cv, groups=groups), 0.1, None)
    scores = abs_res / sig_oof
    n = len(scores)
    q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n)))
    sig_model = _sigma_model().fit(X_tr, abs_res)
    half = q * np.clip(sig_model.predict(X_te), 0.1, None)
    lo, hi = pred_te - half, pred_te + half
    return {"sigma_model": sig_model, "q": q, "level": level,
            "coverage_test": float(np.mean((y_te.values >= lo) & (y_te.values <= hi))),
            "width_test": float(np.mean(hi - lo)),
            "test_low": lo, "test_high": hi}


def train(df: pd.DataFrame, cv_folds: int = 5, verbose: bool = True, include_optional: bool = True) -> TrainResult:
    X, y = df[MODEL_FEATURES], df[TARGET]
    # В датасете ~1 200 строк-дубликатов. Все копии одной пробы держим в одной части выборки,
    # иначе модель «узнаёт» пробу из обучения на проверке и метрики завышаются (утечка данных).
    groups = df.groupby(FEATURES + ["type"], sort=False).ngroup().values
    strata = (df[TARGET].clip(4, 8).astype(str) + df["type"]).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sgkf.split(X, strata, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
    t_te, g_tr = df["type"].iloc[te_idx], groups[tr_idx]
    kf = GroupKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted, oofs = [], {}, {}
    models = candidate_models(include_optional)
    if verbose:
        for lib, err in OPTIONAL_ERRORS.items():
            short = err.splitlines()[0][:160]
            hint = " → на Mac выполните: brew install libomp" if "libomp" in err or "OpenMP" in err else ""
            print(f"  [!] {lib} не подключён: {short}{hint}")
    for name, est in models.items():
        t0 = time.time()
        try:
            oof = cross_val_predict(est, X_tr, y_tr, cv=kf, groups=g_tr, n_jobs=1)
            est.fit(X_tr, y_tr)
        except Exception as e:  # необязательная библиотека несовместима — пропускаем
            if verbose:
                print(f"  {name}: пропущена ({type(e).__name__}: {e})")
            continue
        pred_te = est.predict(X_te)
        m = regression_metrics(y_te.values, pred_te)
        row = {"Модель": name, "R² (CV)": r2_score(y_tr, oof), "MAE (CV)": mean_absolute_error(y_tr, oof),
               **m, **bootstrap_ci(y_te.values, pred_te),
               "R² (обучение)": r2_score(y_tr, est.predict(X_tr)), "Время, с": time.time() - t0}
        rows.append(row)
        fitted[name], oofs[name] = est, oof
        if verbose:
            print(f"  {name:34s} R² CV={row['R² (CV)']:.3f}  R² тест={m['R²']:.3f} "
                  f"[{row['R² ДИ низ']:.2f}–{row['R² ДИ верх']:.2f}]  MAE тест={m['MAE']:.3f}  "
                  f"({row['Время, с']:.0f} с)", flush=True)

    # «Стандартная методика» как в большинстве статей: случайная 5-кратная CV по всем данным без учёта
    # дубликатов. Даёт более высокий R² (копии пробы попадают и в обучение, и в проверку) —
    # считаем для сравнения с чужими результатами, но модель выбираем по строгой оценке.
    rnd = KFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    for row in rows:
        name = row["Модель"]
        if name in STANDARD_CV_MODELS or name in ("XGBoost", "LightGBM", "CatBoost"):
            try:
                p = cross_val_predict(candidate_models(include_optional)[name], X, y, cv=rnd, n_jobs=1)
                row["R² (стандартная CV)"] = r2_score(y, p)
                if verbose:
                    print(f"  стандартная методика · {name:30s} R²={row['R² (стандартная CV)']:.3f}", flush=True)
            except Exception:
                row["R² (стандартная CV)"] = np.nan

    metrics = pd.DataFrame(rows).sort_values("R² (CV)", ascending=False).reset_index(drop=True)
    best_name = metrics.loc[0, "Модель"]  # выбор по кросс-валидации, не по тесту
    best = fitted[best_name]
    pred_te = best.predict(X_te)
    interval = fit_interval(X_tr, y_tr, oofs[best_name], g_tr, kf, X_te, y_te, pred_te)
    if verbose:
        print(f"Лучшая модель по R² на кросс-валидации: {best_name}")
        print(f"Интервал {int(INTERVAL_LEVEL * 100)} %: накрывает факт в {interval['coverage_test']:.1%} "
              f"тестовых проб, средняя ширина {interval['width_test']:.2f} балла")

    ranges = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in MODEL_FEATURES}
    return TrainResult(best, best_name, metrics, X_te, y_te, t_te, pred_te, ranges, interval)


def refit_full(result: TrainResult, df: pd.DataFrame):
    """Переобучает выбранную модель на всех данных для использования в продуктиве."""
    est = candidate_models()[result.model_name]
    est.fit(df[MODEL_FEATURES], df[TARGET])
    return est


def save_model(model, result: TrainResult, path: str | Path, importance: pd.DataFrame | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iv = result.interval
    joblib.dump({
        "version": 3,
        "model": model,
        "model_name": result.model_name,
        "features": MODEL_FEATURES,
        "metrics": result.metrics,
        "feature_ranges": result.feature_ranges,
        "interval": {k: iv[k] for k in ("sigma_model", "q", "level", "coverage_test", "width_test")},
        # данные отложенной выборки — для графиков на вкладке «Модель»
        "test": pd.DataFrame({"y_true": result.y_test.values, "y_pred": result.y_pred,
                              "type": result.test_type.values,
                              "low": iv["test_low"], "high": iv["test_high"]}),
        "importance": importance,
    }, path, compress=3)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу с прогнозом, интервалом и предупреждениями о выходе за диапазон."""
    model = bundle["model"]
    feats = bundle["features"]
    raw = model.predict(X[feats])
    out = X.copy()
    out["predicted_quality"] = np.round(raw, 2)
    out["predicted_score"] = round_score(raw)
    out["quality_band"] = [quality_band(v) for v in raw]

    iv = bundle.get("interval")
    if iv:
        half = iv["q"] * np.clip(iv["sigma_model"].predict(X[feats]), 0.1, None)
        out["ci_low"] = np.round(np.clip(raw - half, 0, 10), 2)
        out["ci_high"] = np.round(np.clip(raw + half, 0, 10), 2)

    ranges = bundle.get("feature_ranges", {})
    warns = []
    for _, row in X[feats].iterrows():
        bad = [c for c in feats if c in ranges and c != "is_red"
               and not (ranges[c][0] <= row[c] <= ranges[c][1])]
        warns.append(("вне диапазона обучения: " + ", ".join(bad)) if bad else "")
    out["warning"] = warns
    return out
