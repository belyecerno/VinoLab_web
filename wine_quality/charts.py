"""Интерактивные графики Plotly в тёмной теме интерфейса."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import FEATURES, RU_SHORT, TARGET

# Палитра тёмной темы: категориальные слоты в фиксированном порядке,
# расходящаяся шкала синий ↔ красный с нейтральной серой серединой, статусные цвета — только для уровня качества.
WHITE_C, RED_C = "#2a78d6", "#eb6834"
POS_C, NEG_C, MID_C = "#2a78d6", "#e34948", "#f0efec"
ACCENT, GOLD = "#a3243f", "#e0a52e"
INK, INK2, MUTED = "#1f1a1d", "#4f464b", "#857b80"
GRID = "#ebe5df"
BAND_C = {"низкое": "#d03b3b", "среднее": "#fab219", "высокое": "#0ca30c"}
TYPE_C = {"white": WHITE_C, "red": RED_C}
TYPE_RU = {"white": "Белое", "red": "Красное"}
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, height: int = 380, title: str | None = None, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=10, r=20, t=50 if title else 20, b=10))
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2, size=13),
        margin=margin,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)) if title else None,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d9d0c8", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#b9aea6", linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#b9aea6", linecolor=GRID)
    return fig


# ---------- Прогноз одной пробы ----------

def gauge(value: float, low: float | None = None, high: float | None = None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(valueformat=".2f", font=dict(size=56, color=INK, family=FONT)),
        gauge=dict(
            axis=dict(range=[3, 9], tickvals=list(range(3, 10)), tickcolor=MUTED, tickfont=dict(color=INK2)),
            bar=dict(color=INK, thickness=0.22),
            bgcolor="#f6f2ee", borderwidth=0,
            steps=[dict(range=[3, 5.5], color="#f19a9a"),
                   dict(range=[5.5, 6.5], color="#fbd36b"),
                   dict(range=[6.5, 9], color="#8fd68f")]
                  + ([dict(range=[max(3, low), min(9, high)], color="#2a78d6", thickness=0.3)]
                     if low is not None else []),
            threshold=dict(line=dict(color="#1c5cab", width=4), thickness=0.95, value=value),
        ),
    ))
    return _layout(fig, height=260, margin=dict(l=30, r=30, t=20, b=0))


def contributions_bar(contrib: pd.DataFrame, top: int = 8) -> go.Figure:
    c = contrib.head(top).iloc[::-1]
    labels = [RU_SHORT[f] for f in c["feature"]]
    colors = [POS_C if v >= 0 else NEG_C for v in c["effect"]]
    hover = [f"<b>{RU_SHORT[f]}</b><br>в пробе: {v:.4g}<br>типично: {t:.4g}<br>влияние: {e:+.2f} балла"
             for f, v, t, e in zip(c["feature"], c["value"], c["typical"], c["effect"])]
    fig = go.Figure(go.Bar(
        x=c["effect"], y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{e:+.2f}" for e in c["effect"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK2), hovertext=hover, hoverinfo="text"))
    lim = max(0.15, float(np.abs(c["effect"]).max()) * 1.35)
    fig.update_xaxes(range=[-lim, lim], title="изменение оценки, баллы", zeroline=True)
    return _layout(fig, height=60 + 36 * len(c), title="Что повлияло на оценку", bargap=0.35)


def radar(pct_sample: pd.Series, pct_ref: pd.Series) -> go.Figure:
    cats = [RU_SHORT[f] for f in FEATURES]
    fig = go.Figure()
    for vals, name, color, dash, fill in (
            (pct_ref, "Эталон (≥7 баллов)", GOLD, "dot", "rgba(242,184,75,0.10)"),
            (pct_sample, "Ваша проба", WHITE_C, "solid", "rgba(57,135,229,0.25)")):
        v = [vals[f] for f in FEATURES]
        fig.add_trace(go.Scatterpolar(
            r=v + v[:1], theta=cats + cats[:1], name=name, fill="toself", fillcolor=fill,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{theta}: %{r:.0f}-й перцентиль<extra>" + name + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], gridcolor=GRID, tickfont=dict(size=10, color=MUTED),
                        linecolor=GRID),
        angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(size=11, color=INK2))))
    return _layout(fig, height=420, title="Профиль пробы (перцентили среди вин того же типа)")


# ---------- Партия проб ----------

def score_histogram(res: pd.DataFrame) -> go.Figure:
    qs = list(range(3, 10))
    cnt = res["predicted_score"].value_counts().reindex(qs, fill_value=0)
    band = ["низкое" if q <= 5 else "среднее" if q == 6 else "высокое" for q in qs]
    fig = go.Figure(go.Bar(
        x=qs, y=cnt.values, marker=dict(color=[BAND_C[b] for b in band], cornerradius=4),
        text=[v if v else "" for v in cnt.values], textposition="outside", cliponaxis=False,
        customdata=band, hovertemplate="Оценка %{x} (%{customdata})<br>%{y} проб<extra></extra>"))
    fig.update_xaxes(tickvals=qs, title="прогноз оценки")
    fig.update_yaxes(title="проб")
    return _layout(fig, height=320, title="Распределение прогнозов", bargap=0.3)


def batch_map(df: pd.DataFrame, res: pd.DataFrame) -> go.Figure:
    """Пробы пользователя на фоне всего датасета: спирт × летучая кислотность — два главных фактора."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["alcohol"], y=df["volatile acidity"], mode="markers", name="датасет",
        marker=dict(size=5, color=df[TARGET], colorscale=[[0, "#e4ddd6"], [1, "#b9aea6"]], opacity=0.35),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=res["alcohol"], y=res["volatile acidity"], mode="markers", name="ваши пробы",
        marker=dict(size=14, color=res["predicted_quality"], cmin=4, cmax=8,
                    colorscale=[[0, NEG_C], [0.5, MID_C], [1, POS_C]],
                    line=dict(color=INK, width=1.5),
                    colorbar=dict(title=dict(text="прогноз"), thickness=10, len=0.8, outlinewidth=0)),
        text=[f"Проба №{i + 1}" for i in range(len(res))],
        hovertemplate="<b>%{text}</b><br>спирт %{x:.1f}%<br>летучая кисл. %{y:.2f}<br>"
                      "прогноз %{marker.color:.2f}<extra></extra>"))
    fig.update_xaxes(title="спирт, % об.")
    fig.update_yaxes(title="летучая кислотность, г/дм³")
    return _layout(fig, height=380, title="Где ваши пробы относительно 6 497 образцов")


# ---------- Аналитика ----------

def correlation_bar(corr: pd.DataFrame, column: str) -> go.Figure:
    c = corr.reindex(corr[column].abs().sort_values().index)
    fig = go.Figure(go.Bar(
        x=c[column], y=c["Показатель"], orientation="h",
        marker=dict(color=[POS_C if v >= 0 else NEG_C for v in c[column]], cornerradius=4),
        text=[f"{v:+.2f}" for v in c[column]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK2),
        hovertemplate="<b>%{y}</b><br>корреляция Спирмена: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(range=[-0.65, 0.65], title="← хуже качество   ·   лучше качество →")
    return _layout(fig, height=460, bargap=0.3)


def importance_bar(imp: pd.DataFrame) -> go.Figure:
    c = imp.iloc[::-1]
    share = c["Рост MAE"] / c["Рост MAE"].clip(lower=0).sum() * 100
    fig = go.Figure(go.Bar(
        x=c["Рост MAE"], y=c["Показатель"], orientation="h",
        error_x=dict(type="data", array=c["Ст. откл."], color=MUTED, thickness=1, width=3),
        marker=dict(color=c["Рост MAE"], colorscale=[[0, "#86b6ef"], [1, "#1c5cab"]], cornerradius=4),
        customdata=share, text=[f"{s:.0f}%" for s in share], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK2),
        hovertemplate="<b>%{y}</b><br>рост ошибки: +%{x:.3f} балла<br>доля важности: %{customdata:.1f}%<extra></extra>"))
    fig.update_xaxes(title="рост ошибки модели при «отключении» признака, баллы")
    return _layout(fig, height=460, bargap=0.3)


def feature_by_quality(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()
    for t in ("white", "red"):
        s = df[df["type"] == t]
        fig.add_trace(go.Box(
            x=s[TARGET], y=s[feature], name=TYPE_RU[t], marker_color=TYPE_C[t],
            boxpoints=False, line=dict(width=1.5), fillcolor=rgba(TYPE_C[t], 0.35)))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(title="оценка качества", tickvals=list(range(3, 10)))
    fig.update_yaxes(title=RU_SHORT[feature])
    return _layout(fig, height=400, boxgap=0.25, boxgroupgap=0.1)


def heatmap(df: pd.DataFrame) -> go.Figure:
    cols = FEATURES + [TARGET]
    m = df[cols].corr(method="spearman")
    labels = [RU_SHORT[c] for c in cols]
    fig = go.Figure(go.Heatmap(
        z=m.values, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, NEG_C], [0.5, MID_C], [1, POS_C]],
        text=np.round(m.values, 2), texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-40)
    return _layout(fig, height=560)


def quality_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t in ("white", "red"):
        cnt = df[df["type"] == t][TARGET].value_counts().sort_index()
        fig.add_trace(go.Bar(x=cnt.index, y=cnt.values, name=TYPE_RU[t],
                             marker=dict(color=TYPE_C[t], cornerradius=4),
                             hovertemplate=TYPE_RU[t] + ": оценка %{x} — %{y} образцов<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="оценка дегустаторов", tickvals=list(range(3, 10)))
    fig.update_yaxes(title="образцов")
    return _layout(fig, height=360, bargap=0.25, bargroupgap=0.08)


# ---------- Модель ----------

def model_comparison(metrics: pd.DataFrame, best: str) -> go.Figure:
    # базовая модель (всегда среднее) — точка отсчёта с R² = 0 по определению; показываем её подписью, не столбиком
    m = metrics[~metrics["Модель"].str.startswith("Базовая")].sort_values("R² (CV)")
    err_hi = m["R² ДИ верх"] - m["R²"]
    err_lo = m["R²"] - m["R² ДИ низ"]
    fig = go.Figure(go.Bar(
        x=m["R²"], y=m["Модель"], orientation="h",
        marker=dict(color=[ACCENT if n == best else "#cfc6c0" for n in m["Модель"]], cornerradius=4),
        error_x=dict(type="data", array=err_hi, arrayminus=err_lo, color=MUTED, thickness=1.2, width=4),
        text=[f"{v:.3f}".replace(".", ",") for v in m["R²"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color=["#ffffff" if n == best else INK for n in m["Модель"]], size=13),
        customdata=np.stack([m["R² ДИ низ"], m["R² ДИ верх"], m["R² (CV)"], m["MAE"]], axis=1),
        hovertemplate="<b>%{y}</b><br>R² на тесте %{x:.3f}<br>95 % ДИ: %{customdata[0]:.3f} – %{customdata[1]:.3f}"
                      "<br>R² на кросс-валидации %{customdata[2]:.3f}<br>MAE %{customdata[3]:.3f}<extra></extra>"))
    fig.update_xaxes(title="R² на тестовой выборке (усы — 95 % доверительный интервал)",
                     range=[min(0, m["R² ДИ низ"].min()) - 0.02, max(0.6, m["R² ДИ верх"].max() + 0.08)])
    return _layout(fig, height=60 + 34 * len(m), bargap=0.35)


def interval_check(test: pd.DataFrame, n: int = 120) -> go.Figure:
    """Случайные пробы теста: интервал прогноза и фактическая оценка — попала ли она внутрь."""
    s = test.sample(min(n, len(test)), random_state=1).sort_values("y_pred").reset_index(drop=True)
    inside = (s["y_true"] >= s["low"]) & (s["y_true"] <= s["high"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.repeat(s.index, 3), y=np.column_stack([s["low"], s["high"], np.full(len(s), np.nan)]).ravel(),
        mode="lines", line=dict(color=rgba(GOLD, 0.55), width=4), name="90 % интервал", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s.index, y=s["y_pred"], mode="markers", name="прогноз",
                             marker=dict(color=INK, size=5), hovertemplate="прогноз %{y:.2f}<extra></extra>"))
    for ok, name, color, sym in ((True, "факт внутри", POS_C, "circle"), (False, "факт снаружи", NEG_C, "x")):
        k = s[inside == ok]
        fig.add_trace(go.Scatter(x=k.index, y=k["y_true"], mode="markers", name=name,
                                 marker=dict(color=color, size=8, symbol=sym),
                                 customdata=np.stack([k["low"], k["high"]], axis=1),
                                 hovertemplate="факт %{y}<br>интервал %{customdata[0]:.2f} – %{customdata[1]:.2f}"
                                               "<extra></extra>"))
    fig.update_xaxes(title="пробы тестовой выборки (по возрастанию прогноза)", showticklabels=False)
    fig.update_yaxes(title="оценка", range=[2.5, 9.5])
    return _layout(fig, height=420)


def predicted_vs_actual(test: pd.DataFrame) -> go.Figure:
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for t in ("white", "red"):
        s = test[test["type"] == t]
        fig.add_trace(go.Scattergl(
            x=s["y_true"] + rng.uniform(-0.2, 0.2, len(s)), y=s["y_pred"], mode="markers", name=TYPE_RU[t],
            marker=dict(size=6, color=TYPE_C[t], opacity=0.55),
            hovertemplate="факт %{x:.0f} · прогноз %{y:.2f}<extra>" + TYPE_RU[t] + "</extra>"))
    fig.add_trace(go.Scatter(x=[2.5, 9.5], y=[2.5, 9.5], mode="lines", name="идеальный прогноз",
                             line=dict(color=MUTED, dash="dash", width=1.5), hoverinfo="skip"))
    fig.update_xaxes(title="фактическая оценка", range=[2.5, 9.5], tickvals=list(range(3, 10)))
    fig.update_yaxes(title="прогноз модели", range=[2.5, 9.5])
    return _layout(fig, height=420)


def confusion(test: pd.DataFrame) -> go.Figure:
    qs = list(range(3, 10))
    pred = np.clip(np.floor(test["y_pred"] + 0.5), 3, 9).astype(int)
    cm = pd.crosstab(pd.Categorical(test["y_true"], qs), pd.Categorical(pred, qs), dropna=False).values
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(
        z=norm, x=qs, y=qs, zmin=0, zmax=1, colorscale=[[0, "#ffffff"], [0.4, "#86b6ef"], [1, "#1c5cab"]],
        text=np.where(cm > 0, cm, ""), texttemplate="%{text}", customdata=cm,
        hovertemplate="факт %{y} → прогноз %{x}<br>%{customdata} проб (%{z:.0%} строки)<extra></extra>",
        showscale=False))
    fig.update_xaxes(title="прогноз (округлённый)", tickvals=qs, showgrid=False)
    fig.update_yaxes(title="факт", tickvals=qs, autorange="reversed", showgrid=False)
    return _layout(fig, height=420)


# ---------- Экономика ----------

def triage_tradeoff(curve: pd.DataFrame, chosen) -> go.Figure:
    """Компромисс: чем выше требуемая точность, тем меньше проб можно разметить без дегустации."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["accuracy"] * 100, y=curve["share"] * 100, mode="lines+markers", name="модель на тесте",
        line=dict(color=WHITE_C, width=2), marker=dict(size=6, color=WHITE_C),
        hovertemplate="точность %{x:.1f}%<br>без дегустации %{y:.1f}% проб<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[chosen["accuracy"] * 100], y=[chosen["share"] * 100], mode="markers+text", name="выбранный режим",
        marker=dict(size=16, color=GOLD, line=dict(color=INK, width=2)),
        text=[f"  {chosen['share']:.0%} проб · точность {chosen['accuracy']:.0%}"], textposition="middle right",
        textfont=dict(color=INK), hoverinfo="skip"))
    fig.update_xaxes(title="точность разметки «брак / норма», %", range=[76, 100])
    fig.update_yaxes(title="проб без полной дегустации, %", range=[0, 105])
    return _layout(fig, height=380)


def monthly_costs(before: float, after: float, fee: float) -> go.Figure:
    cats = ["Сейчас", "С VinoLab"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=[before, after], name="дегустации", marker=dict(color=[MUTED, WHITE_C], cornerradius=4),
                         hovertemplate="%{x}: дегустации %{y:,.0f} ₽<extra></extra>"))
    fig.add_trace(go.Bar(x=cats, y=[0, fee], name="подписка VinoLab", marker=dict(color=GOLD, cornerradius=4),
                         hovertemplate="подписка %{y:,.0f} ₽<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="₽ в месяц", tickformat=",.0f")
    return _layout(fig, height=380, bargap=0.45)


def batch_intervals(res: pd.DataFrame, actual=None) -> go.Figure:
    """Прогноз и 90 % интервал для каждой загруженной пробы — только ваши пробы, без датасета."""
    r = res.reset_index(drop=True)
    labels = [f"Проба {i + 1}" for i in range(len(r))][::-1]
    rr = r.iloc[::-1]
    fig = go.Figure()
    for lo, hi, color in ((3, 5.5, "rgba(208,59,59,0.18)"), (5.5, 6.5, "rgba(250,178,25,0.22)"), (6.5, 9, "rgba(12,163,12,0.18)")):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, line_width=0, layer="below")
    has_ci = "ci_low" in rr
    fig.add_trace(go.Scatter(
        x=rr["predicted_quality"], y=labels, mode="markers", name="прогноз",
        marker=dict(size=14, color=[BAND_C[b] for b in rr["quality_band"]], line=dict(color=INK, width=1)),
        error_x=dict(type="data", symmetric=False,
                     array=(rr["ci_high"] - rr["predicted_quality"]) if has_ci else None,
                     arrayminus=(rr["predicted_quality"] - rr["ci_low"]) if has_ci else None,
                     color="#1c5cab", thickness=3, width=8),
        customdata=np.stack([rr["quality_band"], rr.get("ci_low", rr["predicted_quality"]),
                             rr.get("ci_high", rr["predicted_quality"])], axis=1),
        hovertemplate="<b>%{y}</b><br>прогноз %{x:.2f} (%{customdata[0]})<br>интервал 90 %: "
                      "%{customdata[1]} – %{customdata[2]}<extra></extra>"))
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(actual)[::-1], y=labels, mode="markers", name="настоящая оценка",
            marker=dict(size=14, symbol="diamond", color=INK, line=dict(color="#ffffff", width=1.5)),
            hovertemplate="<b>%{y}</b><br>настоящая оценка %{x}<extra></extra>"))
    fig.update_xaxes(title="оценка качества (зоны: низкое · среднее · высокое)", range=[3, 9], tickvals=list(range(3, 10)))
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=max(260, 70 + 44 * len(r)), showlegend=actual is not None)
