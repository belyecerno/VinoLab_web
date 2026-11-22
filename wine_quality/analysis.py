"""Анализ связи показателей с качеством и построение графиков."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

from .data import FEATURES, MODEL_FEATURES, RU_SHORT, TARGET

# Палитра: категориальные слоты в фиксированном порядке, диверging синий↔красный с серой серединой
C_WHITE, C_RED = "#2a78d6", "#eb6834"          # белое — слот 1, красное — слот 2
C_POS, C_NEG, C_MID = "#2a78d6", "#e34948", "#f0efec"
C_ACCENT = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
TYPE_COLORS = {"white": C_WHITE, "red": C_RED}
TYPE_RU = {"white": "Белое", "red": "Красное"}
DIVERGING = LinearSegmentedColormap.from_list("div", [C_NEG, C_MID, C_POS])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    "legend.frameon": False,
})


def _ru(cols):
    return [RU_SHORT.get(c, c) for c in cols]


# ---------- Таблицы ----------

def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Корреляции Спирмена и Пирсона каждого показателя с оценкой: все вина и по типам."""
    rows = []
    for f in FEATURES:
        r = {"Показатель": RU_SHORT[f], "feature": f,
             "Спирмен (все)": df[f].corr(df[TARGET], method="spearman"),
             "Пирсон (все)": df[f].corr(df[TARGET])}
        for t in ("red", "white"):
            s = df[df["type"] == t]
            r[f"Спирмен ({TYPE_RU[t].lower()})"] = s[f].corr(s[TARGET], method="spearman")
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.reindex(out["Спирмен (все)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def importance_table(model, X_test, y_test, n_repeats: int = 15) -> pd.DataFrame:
    """Перестановочная важность: на сколько растёт MAE, если перемешать признак."""
    pi = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                scoring="neg_mean_absolute_error", random_state=0, n_jobs=-1)
    out = pd.DataFrame({"feature": X_test.columns, "Показатель": _ru(X_test.columns),
                        "Рост MAE": pi.importances_mean, "Ст. откл.": pi.importances_std})
    return out.sort_values("Рост MAE", ascending=False).reset_index(drop=True)


# ---------- Графики (каждая функция возвращает Figure) ----------

def fig_quality_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    qs = np.arange(df[TARGET].min(), df[TARGET].max() + 1)
    w = 0.38
    for i, t in enumerate(("white", "red")):
        cnt = df[df["type"] == t][TARGET].value_counts().reindex(qs, fill_value=0)
        bars = ax.bar(qs + (i - 0.5) * (w + 0.02), cnt.values, width=w, color=TYPE_COLORS[t],
                      label=f"{TYPE_RU[t]} (n={cnt.sum()})")
        for b in bars:
            if b.get_height() > 0:
                ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                            textcoords="offset points")
    ax.set_xticks(qs)
    ax.set_xlabel("Оценка качества (дегустаторы, 0–10)")
    ax.set_ylabel("Количество образцов")
    ax.set_title("Распределение оценок качества")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_correlation_with_quality(corr: pd.DataFrame):
    c = corr.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(c))
    vals = c["Спирмен (все)"].values
    ax.barh(y, vals, color=[C_POS if v > 0 else C_NEG for v in vals], height=0.6)
    ax.scatter(c["Спирмен (белое)"], y, marker="o", s=40, facecolor=SURFACE, edgecolor=INK, zorder=3,
               label="только белые")
    ax.scatter(c["Спирмен (красное)"], y, marker="D", s=36, color=INK, zorder=3, label="только красные")
    sub = c[["Спирмен (все)", "Спирмен (белое)", "Спирмен (красное)"]].values
    for yi, v, row in zip(y, vals, sub):
        edge = np.nanmax(row) if v >= 0 else np.nanmin(row)
        edge = max(edge, 0) if v >= 0 else min(edge, 0)
        ax.annotate(f"{v:+.2f}", (edge, yi), xytext=(9 if v >= 0 else -9, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, c["Показатель"])
    ax.axvline(0, color=INK2, lw=1)
    lim = max(0.6, np.nanmax(np.abs(c[["Спирмен (все)", "Спирмен (белое)", "Спирмен (красное)"]].values)) + 0.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Коэффициент корреляции Спирмена с оценкой")
    ax.set_title("Связь показателей с качеством")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(df):
    cols = FEATURES + [TARGET]
    m = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(m.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), _ru(cols), rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), _ru(cols))
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Корреляция Спирмена")
    ax.set_title("Взаимные корреляции показателей")
    fig.tight_layout()
    return fig


def fig_feature_importance(imp: pd.DataFrame, model_name: str):
    c = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(c))
    ax.barh(y, c["Рост MAE"], xerr=c["Ст. откл."], color=C_ACCENT, height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    for yi, v in zip(y, c["Рост MAE"]):
        ax.annotate(f"{v:.3f}", (max(v, 0), yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, c["Показатель"])
    ax.set_xlabel("Рост средней ошибки (MAE, баллы) при перемешивании признака")
    ax.set_title(f"Важность признаков для модели «{model_name}»")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_top_features_by_quality(df, features: list[str]):
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharex=True)
    axes = np.atleast_1d(axes)
    qs = sorted(df[TARGET].unique())
    for ax, f in zip(axes, features):
        for k, t in enumerate(("white", "red")):
            s = df[df["type"] == t]
            data = [s.loc[s[TARGET] == q, f].values for q in qs]
            pos = np.array(qs) + (k - 0.5) * 0.36
            keep = [i for i, d in enumerate(data) if len(d) > 0]
            bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=0.32,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            for b in bp["boxes"]:
                b.set(facecolor=TYPE_COLORS[t], edgecolor=TYPE_COLORS[t], alpha=0.85)
            for part in ("whiskers", "caps"):
                for l in bp[part]:
                    l.set(color=TYPE_COLORS[t], lw=1)
            for l in bp["medians"]:
                l.set(color=SURFACE, lw=1.5)
        ax.set_title(RU_SHORT[f], fontsize=11)
        ax.set_xticks(qs)
        ax.set_xlabel("Оценка")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLORS[t]) for t in ("white", "red")]
    fig.legend(handles, [TYPE_RU[t] for t in ("white", "red")], loc="upper right", ncol=2)
    fig.suptitle("Ключевые показатели по уровням качества", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_predicted_vs_actual(y_true, y_pred, types, model_name: str):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    rng = np.random.default_rng(0)
    for t in ("white", "red"):
        mk = (types == t).values
        ax.scatter(y_true.values[mk] + rng.uniform(-0.18, 0.18, mk.sum()), y_pred[mk], s=10,
                   alpha=0.45, color=TYPE_COLORS[t], label=TYPE_RU[t], edgecolor="none")
    lo, hi = 2.5, 9.5
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", label="идеальный прогноз")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Фактическая оценка"); ax.set_ylabel("Прогноз модели")
    ax.set_title(f"Прогноз vs факт (тест, {model_name})")
    ax.legend(loc="upper left", markerscale=2)

    # матрица ошибок для округлённого прогноза
    qs = np.arange(3, 10)
    pred_cls = np.clip(np.floor(y_pred + 0.5), 3, 9).astype(int)
    cm = pd.crosstab(pd.Categorical(y_true, qs), pd.Categorical(pred_cls, qs), dropna=False).values
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    blues = LinearSegmentedColormap.from_list("seq", ["#f0f5fc", "#86b6ef", "#2a78d6", "#104281"])
    ax2.imshow(cm_norm, cmap=blues, vmin=0, vmax=1)
    for i in range(len(qs)):
        for j in range(len(qs)):
            if cm[i, j]:
                ax2.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                         color="white" if cm_norm[i, j] > 0.5 else INK)
    ax2.set_xticks(range(len(qs)), qs); ax2.set_yticks(range(len(qs)), qs)
    ax2.set_xlabel("Прогноз (округлённый)"); ax2.set_ylabel("Факт")
    ax2.set_title("Матрица ошибок (число образцов)")
    ax2.grid(False)
    fig.tight_layout()
    return fig


def fig_model_comparison(metrics: pd.DataFrame):
    m = metrics.sort_values("R² (CV)")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(m))
    best = metrics.loc[0, "Модель"]
    ax.barh(y, m["MAE"], height=0.6, color=[C_ACCENT if n == best else "#b7d3f6" for n in m["Модель"]])
    for yi, (v, acc) in enumerate(zip(m["MAE"], m["Точность (±1 балл)"])):
        ax.annotate(f"MAE {v:.3f} · ±1 балл: {acc:.0%}", (v, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, m["Модель"])
    ax.set_xlim(0, m["MAE"].max() * 1.45)
    ax.set_xlabel("Средняя абсолютная ошибка на тесте, баллы (меньше — лучше)")
    ax.set_title("Сравнение моделей")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------- Полный отчёт ----------

def build_report(df, result, out_dir: str | Path = "reports") -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    corr = correlation_table(df)
    imp = importance_table(result.model, result.X_test, result.y_test)
    top = [f for f in imp["feature"] if f != "is_red"][:4]

    figs = {
        "01_quality_distribution.png": fig_quality_distribution(df),
        "02_correlation_with_quality.png": fig_correlation_with_quality(corr),
        "03_correlation_heatmap.png": fig_correlation_heatmap(df),
        "04_feature_importance.png": fig_feature_importance(imp, result.model_name),
        "05_top_features_by_quality.png": fig_top_features_by_quality(df, top),
        "06_predicted_vs_actual.png": fig_predicted_vs_actual(
            result.y_test, result.y_pred, result.test_type, result.model_name),
        "07_model_comparison.png": fig_model_comparison(result.metrics),
    }
    for name, fig in figs.items():
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    corr.drop(columns="feature").to_csv(out_dir / "correlations.csv", index=False, float_format="%.3f")
    imp.drop(columns="feature").to_csv(out_dir / "feature_importance.csv", index=False, float_format="%.4f")
    result.metrics.to_csv(out_dir / "model_comparison.csv", index=False, float_format="%.4f")
    _write_markdown(df, result, corr, imp, out_dir)
    return {"correlations": corr, "importance": imp, "figures": list(figs)}


def _fmt_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _write_markdown(df, result, corr, imp, out_dir: Path):
    best = result.metrics.iloc[0]
    pos = corr[corr["Спирмен (все)"] > 0].head(3)["Показатель"].tolist()
    neg = corr[corr["Спирмен (все)"] < 0].head(3)["Показатель"].tolist()
    text = f"""# Отчёт: прогноз оценки качества вина

Данные: {len(df)} образцов ({(df.type == 'white').sum()} белых, {(df.type == 'red').sum()} красных),
UCI Wine Quality. Оценка — медиана оценок дегустаторов по шкале 0–10.

## Модель
Выбрана по кросс-валидации: **{result.model_name}**.
На отложенной выборке (20 %): MAE = {best['MAE']:.3f} балла, RMSE = {best['RMSE']:.3f}, R² = {best['R²']:.3f},
точное попадание в оценку — {best['Точность (точное совпадение)']:.0%}, ошибка не более 1 балла — {best['Точность (±1 балл)']:.0%}.

{_fmt_table(result.metrics)}

## Какие показатели сильнее всего связаны с качеством
Положительная связь (больше — выше оценка): {', '.join(pos)}.
Отрицательная связь (больше — ниже оценка): {', '.join(neg)}.

{_fmt_table(corr.drop(columns='feature'))}

## Важность признаков для модели (перестановочная)
{_fmt_table(imp.drop(columns='feature'))}

Корреляция показывает монотонную связь показателя с оценкой «в одиночку»;
перестановочная важность — насколько модель опирается на признак с учётом остальных.
Оба метода описывают связь, а не причинность.

## Графики
""" + "\n".join(f"![{n}](figures/{n})" for n in sorted(p.name for p in (out_dir / 'figures').glob('*.png')))
    (out_dir / "report.md").write_text(text, encoding="utf-8")
