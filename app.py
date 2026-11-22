"""VinoLab — веб-интерфейс прогноза качества вина.  Запуск: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wine_quality import charts as ch
from wine_quality.analysis import build_report, correlation_table
from wine_quality.data import FEATURES, RU_NAMES, RU_SHORT, load_dataset, prepare_input, read_user_file
from wine_quality.insights import (contributions, dataset_predictions, percentiles, pick_threshold, presets,
                                   rank_among, reference_profile, slider_specs, triage_curve)
from wine_quality.model import load_model, predict, quality_band, refit_full, save_model, train

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "wine_quality_model.joblib"
REPORTS = ROOT / "reports"
PLOT_CFG = {"displayModeBar": False}

st.set_page_config(page_title="VinoLab · Оценка качества вина", page_icon="🍷", layout="wide",
                   initial_sidebar_state="collapsed")

# ============================== Стили ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, button, input, textarea { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
.stApp { background: radial-gradient(1100px 480px at 0% -10%, rgba(163,36,63,.08), transparent 60%), #faf8f5; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }
h1, h2, h3 { letter-spacing: -.02em; color:#1f1a1d; }

.hero { display:flex; align-items:center; justify-content:space-between; gap:24px; margin-bottom: 8px; }
.hero-title { font-size: 40px; font-weight: 800; line-height: 1.05; margin: 0; color:#1f1a1d; }
.hero-title span { color:#a3243f; }
.hero-sub { color:#5f565b; font-size: 16px; margin-top: 8px; }
.chip { display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
  background:#ffffff; border:1px solid #e4dcd4; color:#3b3337; font-size:14px; white-space:nowrap; }
.dot { width:8px; height:8px; border-radius:50%; background:#0ca30c; }

[class*="st-key-card"] {
  background:#ffffff; border: 1px solid #e8e1da; border-radius: 20px; padding: 18px 20px 14px;
  box-shadow: 0 4px 18px rgba(60,40,30,.06);
}
.kpi { background:#ffffff; border:1px solid #e8e1da; border-radius:16px; padding: 14px 16px; height: 100%; }
.kpi-label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color:#7a7075; font-weight:600; }
.kpi-value { font-size: 30px; font-weight: 700; color:#1f1a1d; margin-top: 4px; line-height:1.1; }
.kpi-sub { font-size: 13.5px; color:#5f565b; margin-top: 4px; }
.badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:999px; font-size:15px; font-weight:600; }
.section { font-size: 13px; text-transform: uppercase; letter-spacing:.09em; color:#7a7075; font-weight:700; margin: 6px 0 2px; }
.insight { padding: 10px 12px; border-radius: 12px; background:#f8f4f0;
  border-left: 4px solid var(--c); margin-bottom: 8px; color:#2b2427; font-size: 15px; }
.muted { color:#6b6266; font-size: 14px; }
div[data-testid="stSegmentedControl"] button, div[data-testid="stPills"] button { border-radius: 999px !important; }
div[data-testid="stFileUploaderDropzone"] { border-radius: 16px; background:#fdfbf9; border: 1.5px dashed #d6ccc3; }
</style>
""", unsafe_allow_html=True)

BAND_ICON = {"низкое": "▼", "среднее": "●", "высокое": "▲"}


def kpi(label, value, sub="", accent=None):
    style = f"border-top: 2px solid {accent};" if accent else ""
    return (f'<div class="kpi" style="{style}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    for col, item in zip(st.columns(len(items)), items):
        col.markdown(kpi(*item), unsafe_allow_html=True)


def badge(band):
    c = ch.BAND_C[band]
    return (f'<span class="badge" style="background:{ch.rgba(c, .12)};color:#1f1a1d;border:1px solid {ch.rgba(c, .6)}">'
            f'{BAND_ICON[band]} {band}</span>')


# ============================== Данные и модель ==============================
@st.cache_data(show_spinner="Загрузка датасета…")
def get_data():
    return load_dataset(ROOT / "data")


@st.cache_resource(show_spinner="Обучение модели — 1–3 минуты при первом запуске…")
def get_model():
    # Для веб-версии модель лежит на GitHub частями (лимит размера файла) — склеиваем при первом запуске
    parts = sorted(MODEL_PATH.parent.glob(MODEL_PATH.name + ".part*"))
    if not MODEL_PATH.exists() and parts:
        with open(MODEL_PATH, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
    bundle = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
    if bundle is None or bundle.get("version", 0) < 3:      # нет модели или она от старой версии
        df = get_data()
        result = train(df, verbose=False)
        report = build_report(df, result, REPORTS)
        save_model(refit_full(result, df), result, MODEL_PATH, importance=report["importance"])
        bundle = load_model(MODEL_PATH)
    return bundle


@st.cache_data(show_spinner=False)
def get_static(_bundle_name):
    df = get_data()
    return (slider_specs(df), presets(df), correlation_table(df),
            dataset_predictions(BUNDLE["model"], df))


df = get_data()
BUNDLE = get_model()
SPECS, PRESETS, CORR, ALL_PREDS = get_static(BUNDLE["model_name"])
best_row = BUNDLE["metrics"].iloc[0]

# ============================== Шапка и навигация ==============================
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-title">Vino<span>Lab</span> · оценка качества вина</div>
    <div class="hero-sub">Прогноз экспертной оценки по лабораторным показателям · {len(df):,} образцов UCI Wine Quality</div>
  </div>
  <div class="chip"><span class="dot"></span>{BUNDLE['model_name']} · R² {best_row['R²']:.2f} · ошибка {best_row['MAE']:.2f} балла</div>
</div>
""".replace(",", " "), unsafe_allow_html=True)

PAGES = ["🔮 Прогноз пробы", "📦 Партия проб", "📊 Аналитика", "🧠 Модель", "💰 Экономика"]
page = st.segmented_control("Раздел", PAGES, default=PAGES[0], key="page", label_visibility="collapsed") or PAGES[0]
st.write("")


# ============================== Общие блоки ==============================
def render_explanation(x: pd.Series, res_row: pd.Series, key_prefix: str):
    """Спидометр, KPI, вклад показателей, радар и текстовые выводы для одной пробы."""
    wine_type = "red" if int(x["is_red"]) == 1 else "white"
    pred = float(res_row["predicted_quality"])
    band = res_row["quality_band"]
    contrib = contributions(BUNDLE["model"], x, df)
    better = rank_among(pred, ALL_PREDS)

    with st.container(key=f"card_{key_prefix}_gauge"):
        lo, hi = res_row.get("ci_low"), res_row.get("ci_high")
        st.plotly_chart(ch.gauge(pred, lo, hi), config=PLOT_CFG, width="stretch", key=f"{key_prefix}_g")
        st.markdown(f'<div style="text-align:center;margin-top:-18px">{badge(band)}</div>', unsafe_allow_html=True)
        st.write("")
        level = int(BUNDLE["interval"]["level"] * 100)
        kpi_row([
            ("Оценка", int(res_row["predicted_score"]), "из 10, округлённо", ch.BAND_C[band]),
            (f"Интервал {level} %", f"{lo:.1f} – {hi:.1f}", "истинная оценка в этих пределах", ch.GOLD),
            ("Выше, чем у", f"{better:.0f}%", "образцов датасета", ch.WHITE_C),
        ])
        st.caption(f"С вероятностью {level} % экспертная оценка пробы лежит в интервале {lo:.2f} – {hi:.2f} "
                   f"(на тестовой выборке интервал накрыл факт в {BUNDLE['interval']['coverage_test']:.0%} проб). "
                   "Синяя полоса на шкале — этот интервал.")
        if res_row.get("warning"):
            st.warning("Некоторые значения " + res_row["warning"] + " — прогноз менее надёжен.", icon="⚠️")

    # Текстовые выводы: что тянет вверх и вниз
    with st.container(key=f"card_{key_prefix}_ins"):
        st.markdown('<div class="section">Выводы</div>', unsafe_allow_html=True)
        ups = contrib[contrib["effect"] > 0.02].head(2)
        downs = contrib[contrib["effect"] < -0.02].head(3)
        for _, r in downs.iterrows():
            st.markdown(f'<div class="insight" style="--c:{ch.NEG_C}">▼ <b>{RU_SHORT[r.feature]}</b>: '
                        f'{r.value:.4g} при типичном {r.typical:.4g} — снижает оценку на {abs(r.effect):.2f}</div>',
                        unsafe_allow_html=True)
        for _, r in ups.iterrows():
            st.markdown(f'<div class="insight" style="--c:{ch.POS_C}">▲ <b>{RU_SHORT[r.feature]}</b>: '
                        f'{r.value:.4g} при типичном {r.typical:.4g} — повышает оценку на {r.effect:.2f}</div>',
                        unsafe_allow_html=True)
        if downs.empty and ups.empty:
            st.markdown('<div class="muted">Проба близка к типичной — ни один показатель заметно не смещает оценку.</div>',
                        unsafe_allow_html=True)
    return contrib, wine_type


def render_profile(x, contrib, wine_type, key_prefix):
    with st.container(key=f"card_{key_prefix}_contrib"):
        st.plotly_chart(ch.contributions_bar(contrib), config=PLOT_CFG, width="stretch", key=f"{key_prefix}_c")
        st.caption("Насколько изменится прогноз, если заменить показатель типичным значением для этого типа вина.")


# ============================== 1. Прогноз пробы ==============================
GROUPS = {
    "🧪 Кислотность": ["fixed acidity", "volatile acidity", "citric acid", "pH"],
    "🍯 Сахар и спирт": ["alcohol", "residual sugar", "density"],
    "🛡️ Консерванты и соли": ["free sulfur dioxide", "total sulfur dioxide", "sulphates", "chlorides"],
}


def apply_preset(name):
    p = PRESETS[name]
    st.session_state["wine_type"] = "Красное" if p["type"] == "red" else "Белое"
    for f in FEATURES:
        s = SPECS[f]
        st.session_state[f"v_{f}"] = float(round(p[f], s["nd"]))


def on_preset():
    name = st.session_state.get("preset_pick")
    if name:
        apply_preset(name)
    st.session_state["preset_pick"] = None


# Канонические значения показателей хранятся в v_<признак>; ползунки (sl_) и поля ввода (num_)
# синхронизируются с ними, поэтому можно переключаться между способами ввода без потери значений.
if "v_alcohol" not in st.session_state:
    apply_preset("Типичное белое")
for k in ("wine_type", "input_mode"):
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]


def sync(src, f):
    st.session_state[f"v_{f}"] = float(st.session_state[f"{src}_{f}"])


@st.cache_data(show_spinner=False)
def typical_ranges():
    """Типичный диапазон (5–95 перцентиль) каждого показателя по типу вина."""
    return {t: df[df["type"] == t][FEATURES].quantile([0.05, 0.95]) for t in ("white", "red")}


TYP = typical_ranges()

JOURNAL = ROOT / "saved_samples.csv"


def render_journal(vals: dict, r: pd.Series):
    """Журнал проб: сохраняет введённые показатели и прогноз в saved_samples.csv в папке программы."""
    with st.container(key="card_journal"):
        st.markdown('<div class="section">💾 Журнал проб</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            name = st.text_input("Название пробы", placeholder="например: Шардоне, партия 12, бочка 3",
                                 key="journal_name")
        with c2:
            save = st.button("Сохранить пробу", type="primary", use_container_width=True)
        if save:
            row = {"name": name.strip() or f"Проба {pd.Timestamp.now():%d.%m %H:%M}",
                   "saved_at": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}", **vals,
                   "predicted_quality": r["predicted_quality"], "ci_low": r.get("ci_low"),
                   "ci_high": r.get("ci_high"), "quality_band": r["quality_band"]}
            old = pd.read_csv(JOURNAL, sep=";") if JOURNAL.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
                JOURNAL, sep=";", index=False, encoding="utf-8-sig")
            st.success(f"Сохранено: {row['name']}")
        if JOURNAL.exists():
            j = pd.read_csv(JOURNAL, sep=";")
            st.dataframe(j[["name", "saved_at", "type", "alcohol", "volatile acidity",
                            "predicted_quality", "ci_low", "ci_high", "quality_band"]].rename(columns={
                "name": "Проба", "saved_at": "Когда", "type": "Тип", "alcohol": "Спирт, %",
                "volatile acidity": "Летучая кисл.", "predicted_quality": "Прогноз",
                "ci_low": "от", "ci_high": "до", "quality_band": "Уровень"}),
                hide_index=True, use_container_width=True)
            d1, d2 = st.columns([3, 1])
            with d1:
                st.download_button("⬇ Скачать журнал (CSV, открывается в Excel)", JOURNAL.read_bytes(),
                                   "vinolab_journal.csv", "text/csv")
            with d2:
                if st.button("Очистить журнал", use_container_width=True):
                    JOURNAL.unlink()
                    st.rerun()
            st.caption(f"Журнал хранится в файле {JOURNAL.name} в папке программы и не пропадает после "
                       "перезапуска. Этот файл можно загрузить во вкладку «Партия проб → Свой файл».")


if page == PAGES[0]:
    st.pills("Готовые профили", list(PRESETS), key="preset_pick", on_change=on_preset,
             label_visibility="collapsed")
    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(key="card_inputs"):
        st.markdown('<div class="section">Лабораторные показатели</div>', unsafe_allow_html=True)
        c_t, c_m = st.columns(2)
        with c_t:
            st.segmented_control("Тип вина", ["Белое", "Красное"], key="wine_type")
        with c_m:
            mode = st.segmented_control("Способ ввода", ["Ползунки", "Вручную"], default="Ползунки",
                                        key="input_mode") or "Ползунки"
        cur_type = "red" if st.session_state.get("wine_type") == "Красное" else "white"
        tabs = st.tabs(list(GROUPS))
        for tab, feats in zip(tabs, GROUPS.values()):
            with tab:
                for f in feats:
                    s, v = SPECS[f], st.session_state[f"v_{f}"]
                    t_lo, t_hi = TYP[cur_type][f].values
                    fmt = lambda z: f"{z:.{s['nd']}f}"
                    hint = (f"Типичный диапазон для {'красных' if cur_type == 'red' else 'белых'} вин "
                            f"(90 % проб): {fmt(t_lo)} – {fmt(t_hi)}")
                    if mode == "Ползунки":
                        st.session_state[f"sl_{f}"] = float(np.clip(v, s["min"], s["max"]))
                        st.slider(RU_NAMES[f], min_value=s["min"], max_value=s["max"], step=s["step"],
                                  format=s["format"], key=f"sl_{f}", help=hint, on_change=sync, args=("sl", f))
                    else:
                        st.session_state[f"num_{f}"] = float(v)
                        st.number_input(RU_NAMES[f], min_value=0.0, max_value=float(s["max"] * 3), step=s["step"],
                                        format=s["format"], key=f"num_{f}", help=hint, on_change=sync, args=("num", f))
                    flag = "" if t_lo <= v <= t_hi else " · <span style='color:#fab219'>вне типичного</span>"
                    st.markdown(f'<div class="muted" style="margin:-10px 0 8px">норма {fmt(t_lo)} – {fmt(t_hi)}'
                                f'{flag}</div>', unsafe_allow_html=True)
        st.caption("Ползунки охватывают 99 % значений датасета; в режиме «Вручную» можно ввести любое значение "
                   "с клавиатуры. Прогноз пересчитывается сразу.")

    vals = {f: st.session_state[f"v_{f}"] for f in FEATURES}
    vals["type"] = "red" if st.session_state.get("wine_type") == "Красное" else "white"
    X = prepare_input(pd.DataFrame([vals]))
    res = predict(BUNDLE, X)
    with right:
        contrib, wtype = render_explanation(X.iloc[0], res.iloc[0], "single")
    render_profile(X.iloc[0], contrib, wtype, "single")
    render_journal(vals, res.iloc[0])

# ============================== 2. Партия проб ==============================
elif page == PAGES[1]:
    SRC = ["📄 Свой файл", "🧪 Пример: 4 новые пробы", "✅ Проверка: 10 проб, которых модель не видела"]
    with st.container(key="card_upload"):
        src = st.segmented_control("Откуда взять пробы", SRC, default=SRC[1], key="batch_src") or SRC[1]
        st.markdown(
            f'<div class="muted" style="margin:4px 0 8px">Модель обучена на {len(df):,} винах с известной оценкой. '
            'Здесь она оценивает <b>новые</b> пробы — так ей будут пользоваться на винодельне.</div>'.replace(",", " "),
            unsafe_allow_html=True)
        up = None
        if src == SRC[0]:
            c1, c2 = st.columns([2.2, 1], gap="large")
            with c1:
                up = st.file_uploader("Файл с результатами измерений (CSV или Excel)",
                                      type=["csv", "txt", "xlsx", "xls"])
            with c2:
                default_type = st.segmented_control("Тип, если нет колонки type", ["Белое", "Красное"],
                                                    default="Белое", key="batch_type") or "Белое"
                tpl = pd.DataFrame([PRESETS["Типичное белое"]])[FEATURES + ["type"]]
                st.download_button("⬇ Шаблон CSV", tpl.to_csv(index=False, sep=";"), "template.csv", "text/csv")
        elif src == SRC[1]:
            st.caption("4 пробы из файла samples_template.csv: замеры вин, оценку которых модель не знает.")
        else:
            st.caption("Случайные пробы из проверочной части данных: модель их не видела при обучении, "
                       "а настоящая оценка дегустаторов известна — можно сравнить.")
            if st.button("🔄 Другие 10 проб"):
                st.session_state["check_seed"] = st.session_state.get("check_seed", 0) + 1

    X, res, actual = None, None, None
    try:
        if src == SRC[0] and up is not None:
            X = read_user_file(up, default_type="red" if default_type == "Красное" else "white")
        elif src == SRC[1] and (ROOT / "samples_template.csv").exists():
            X = read_user_file(ROOT / "samples_template.csv")
    except ValueError as e:
        st.error(f"Не удалось прочитать файл: {e}", icon="🚫")
    if X is not None:
        res = predict(BUNDLE, X)
    elif src == SRC[2]:
        t = BUNDLE["test"].sample(10, random_state=st.session_state.get("check_seed", 0)).reset_index(drop=True)
        res = pd.DataFrame({
            "is_red": (t["type"] == "red").astype(int), "predicted_quality": t["y_pred"].round(2),
            "ci_low": t["low"].round(2), "ci_high": t["high"].round(2), "warning": "",
            "quality_band": [quality_band(v) for v in t["y_pred"]]})
        actual = t["y_true"].values

    if res is not None:
        bands = res["quality_band"].value_counts()
        n = len(res)
        st.write("")
        if actual is None:
            kpi_row([
                ("Новых проб", n, "прогнозы ниже — только по ним", None),
                ("Средний прогноз", f"{res['predicted_quality'].mean():.2f}", "из 10 баллов", ch.GOLD),
                ("Высокое качество", f"{bands.get('высокое', 0)} из {n}", "оценка 7 и выше", ch.BAND_C["высокое"]),
                ("Низкое качество", f"{bands.get('низкое', 0)} из {n}", "оценка 5 и ниже", ch.BAND_C["низкое"]),
            ])
        else:
            inside = int(((actual >= res["ci_low"]) & (actual <= res["ci_high"])).sum())
            err = float(np.abs(res["predicted_quality"] - actual).mean())
            near = int((np.abs(np.floor(res["predicted_quality"] + 0.5) - actual) <= 1).sum())
            kpi_row([
                ("Проб для проверки", n, "модель их не видела", None),
                ("Средняя ошибка", f"±{err:.2f}", "балла на этих пробах", ch.ACCENT),
                ("Ошибка не больше 1 балла", f"{near} из {n}", "проб", ch.BAND_C["высокое"]),
                ("Оценка внутри интервала", f"{inside} из {n}", "проб (ожидаем около 9 из 10)", ch.WHITE_C),
            ])
        st.write("")
        with st.container(key="card_batch_chart"):
            st.markdown(f'<div class="section">Прогноз и интервал 90 % для каждой из {n} проб</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ch.batch_intervals(res, actual), config=PLOT_CFG, width="stretch")
            st.caption("Круг — прогноз, синяя линия — интервал, в котором с вероятностью 90 % лежит оценка дегустаторов"
                       + ("; ромб — настоящая оценка." if actual is not None else "."))

        with st.container(key="card_table"):
            st.markdown('<div class="section">Результаты</div>', unsafe_allow_html=True)
            show = pd.DataFrame({
                "Проба": [f"№ {i + 1}" for i in range(n)],
                "Тип": np.where(res["is_red"] == 1, "красное", "белое"),
                "Прогноз": res["predicted_quality"],
                "Интервал 90 %": [f"{lo:.1f} – {hi:.1f}" for lo, hi in zip(res["ci_low"], res["ci_high"])],
                "Уровень": res["quality_band"],
                "Примечание": res["warning"],
            })
            if actual is not None:
                show.insert(4, "Настоящая оценка", actual)
                show["Примечание"] = np.where((actual >= res["ci_low"]) & (actual <= res["ci_high"]),
                                              "внутри интервала", "вне интервала")
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                "Прогноз": st.column_config.ProgressColumn("Прогноз", min_value=3.0, max_value=9.0, format="%.2f"),
            })
            st.download_button("⬇ Скачать результаты (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                               "predictions.csv", "text/csv", type="primary")

        if X is not None:
            with st.expander("Подробно: почему у пробы такая оценка"):
                pick = st.selectbox("Проба", range(n), format_func=lambda i: f"Проба №{i + 1} — "
                                    f"{res.iloc[i]['predicted_quality']:.2f} ({res.iloc[i]['quality_band']})")
                a, b = st.columns([1, 1], gap="large")
                with a:
                    contrib, wtype = render_explanation(X.iloc[pick], res.iloc[pick], "batch")
                with b, st.container(key="card_batch_contrib"):
                    st.plotly_chart(ch.contributions_bar(contrib), config=PLOT_CFG, width="stretch", key="batch_c")
    else:
        st.info("Загрузите файл с замерами. Колонки: " + ", ".join(FEATURES) +
                "; необязательно — type (red/white). Шаблон — кнопка выше.", icon="📄")

# ============================== 3. Аналитика ==============================
elif page == PAGES[2]:
    top = CORR.iloc[0]
    neg = CORR[CORR["Спирмен (все)"] < 0].iloc[0]
    imp = BUNDLE.get("importance")
    kpi_row([
        ("Образцов в данных", f"{len(df):,}".replace(",", " "),
         f"{(df.type == 'white').sum()} белых · {(df.type == 'red').sum()} красных", None),
        ("Сильнее всего повышает оценку", top["Показатель"], f"корреляция {top['Спирмен (все)']:+.2f}", ch.POS_C),
        ("Сильнее всего снижает оценку", neg["Показатель"], f"корреляция {neg['Спирмен (все)']:+.2f}", ch.NEG_C),
    ])
    st.write("")
    with st.container(key="card_corr"):
        st.markdown('<div class="section">Какие показатели связаны с качеством</div>', unsafe_allow_html=True)
        which = st.segmented_control("Вина", ["Все", "Белые", "Красные"], default="Все", key="corr_which",
                                     label_visibility="collapsed") or "Все"
        col = {"Все": "Спирмен (все)", "Белые": "Спирмен (белое)", "Красные": "Спирмен (красное)"}[which]
        st.plotly_chart(ch.correlation_bar(CORR, col), config=PLOT_CFG, width="stretch")
        st.caption("Синий столбик вправо — чем больше показателя, тем выше оценка; красный влево — тем ниже. "
                   "Чем длиннее столбик, тем сильнее связь (корреляция Спирмена).")

    with st.container(key="card_explorer"):
        c1, c2 = st.columns([1, 3], gap="large")
        with c1:
            st.markdown('<div class="section">Посмотреть показатель</div>', unsafe_allow_html=True)
            order = CORR["feature"].tolist()
            feat = st.selectbox("Показатель", order, format_func=lambda f: RU_SHORT[f])
            hi, lo = df[df.quality >= 7][feat].median(), df[df.quality <= 4][feat].median()
            st.markdown(kpi("Типичное значение", f"{hi:.3g} / {lo:.3g}", "у хороших (≥7) / плохих (≤4) вин"),
                        unsafe_allow_html=True)
        with c2:
            st.plotly_chart(ch.feature_by_quality(df, feat), config=PLOT_CFG, width="stretch")

    with st.expander("Подробнее: важность для модели, распределение оценок, взаимные корреляции"):
        if imp is None and (REPORTS / "feature_importance.csv").exists():
            imp = pd.read_csv(REPORTS / "feature_importance.csv")
        if imp is not None:
            st.markdown('<div class="section">Важность признаков для модели</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.importance_bar(imp), config=PLOT_CFG, width="stretch")
        c1, c2 = st.columns([1, 1.25], gap="medium")
        with c1:
            st.markdown('<div class="section">Распределение оценок</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.quality_distribution(df), config=PLOT_CFG, width="stretch")
        with c2:
            st.markdown('<div class="section">Взаимные корреляции показателей</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.heatmap(df), config=PLOT_CFG, width="stretch")

# ============================== 4. Модель ==============================
elif page == PAGES[3]:
    m, iv = best_row, BUNDLE["interval"]
    mets = BUNDLE["metrics"]
    std_col = "R² (стандартная CV)"
    base = mets[mets["Модель"].str.startswith("Базовая")]
    base_mae = float(base["MAE"].iloc[0]) if len(base) else None
    std_r2 = float(mets[std_col].max()) if std_col in mets and mets[std_col].notna().any() else None
    kpi_row([
        ("Средняя ошибка", f"±{m['MAE']:.2f} балла", "по 10-балльной шкале", ch.ACCENT),
        ("Ошибка не больше 1 балла", f"{m['Точность (±1 балл)']:.0%}", "прогнозов на новых пробах", ch.BAND_C["высокое"]),
        ("R² честный", f"{m['R²']:.2f}", (f"{std_r2:.2f} по стандартной методике" if std_r2 else "на новых пробах"),
         ch.GOLD),
    ])
    st.write("")
    with st.container(key="card_cmp"):
        st.markdown(f'<div class="section">Сравнение {len(mets) - len(base)} моделей · лучшая — {BUNDLE["model_name"]}'
                    f'</div>', unsafe_allow_html=True)
        st.plotly_chart(ch.model_comparison(mets, BUNDLE["model_name"]), config=PLOT_CFG, width="stretch")
        st.caption("R² показывает, насколько модель лучше, чем всегда называть среднюю оценку. У такой «средней» "
                   "модели R² = 0 по определению — это точка отсчёта"
                   + (f" (её ошибка ±{base_mae:.2f} балла)." if base_mae else ".")
                   + " Усы — 95 % доверительный интервал.")

    with st.container(key="card_r2why"):
        tr = m["R² (обучение)"]
        st.markdown(f"""<div class="section">Почему R² не близок к 1</div>
<div class="insight" style="--c:{ch.GOLD}"><b>Оценку ставят люди.</b> Вкус, аромат и сорт в лабораторных данных не измерены,
поэтому часть разброса оценок химией не объясняется — у всех моделей один потолок.</div>
<div class="insight" style="--c:{ch.NEG_C}"><b>Высокий R² бывает обманчив.</b> На обучающих данных у нас R² = {tr:.2f}, но это
заучивание. Мы показываем R² на пробах, которых модель не видела.</div>""", unsafe_allow_html=True)

    with st.expander("Подробнее: проверка модели и интервала"):
        kpi_row([
            ("R² на обучающих данных", f"{m['R² (обучение)']:.2f}", "заучивание, не качество", ch.NEG_C),
            (f"Интервал {int(iv['level'] * 100)} %", f"{iv['coverage_test']:.1%}",
             f"проб накрыто · ширина ≈ {iv['width_test']:.1f} балла", ch.WHITE_C),
            ("Точное попадание", f"{m['Точность (точное совпадение)']:.0%}", "округлённый прогноз = оценка", None),
        ])
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="section">Прогноз против факта</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.predicted_vs_actual(BUNDLE["test"]), config=PLOT_CFG, width="stretch")
        with c2:
            st.markdown('<div class="section">Проверка интервала на тестовых пробах</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.interval_check(BUNDLE["test"]), config=PLOT_CFG, width="stretch")
        st.markdown('<div class="section">Матрица ошибок</div>', unsafe_allow_html=True)
        st.plotly_chart(ch.confusion(BUNDLE["test"]), config=PLOT_CFG, width="stretch")
        st.dataframe(mets.round(3), width="stretch", hide_index=True)
        st.markdown('<div class="muted">Модель выбирается по R² на 5-кратной кросс-валидации (80 % данных), '
                    'проверяется на отложенных 20 %, затем переобучается на всех данных. Дубликаты проб всегда в одной '
                    'части выборки. Интервал — адаптивное конформное предсказание.</div>', unsafe_allow_html=True)

# ============================== 5. Экономика ==============================
else:
    curve = triage_curve(BUNDLE["test"])
    st.markdown('<div class="muted" style="margin-bottom:10px">Сценарий: VinoLab оценивает каждую пробу сразу после '
                'лабораторного анализа. Если модель уверена, что проба — «брак» (≤5) или «норма» (≥6), полная '
                'дегустация не нужна; спорные пробы идут дегустаторам. Доля уверенных проб и их точность взяты '
                'из проверки модели на тестовых пробах, которых она не видела.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.6], gap="large")
    with left, st.container(key="card_econ_in"):
        st.markdown('<div class="section">Параметры предприятия</div>', unsafe_allow_html=True)
        n = st.number_input("Проб в месяц", min_value=1, max_value=100000, value=200, step=10)
        cost = st.number_input("Стоимость одной дегустационной оценки, ₽", min_value=0, value=3000, step=500,
                               help="Работа комиссии, время экспертов, образцы. Замените на свою цифру.")
        fee = st.number_input("Подписка VinoLab, ₽ в месяц", min_value=0, value=15000, step=1000)
        setup = st.number_input("Внедрение (разово), ₽", min_value=0, value=100000, step=10000,
                                help="Переобучение модели на данных предприятия и интеграция с лабораторией.")
        target = st.slider("Требуемая точность разметки, %", 85, 96, 92, step=1)
        st.caption("Значения по умолчанию — допущения для примера; подставьте данные своего производства.")
    chosen = pick_threshold(curve, target / 100)
    skipped = n * chosen["share"]
    before = n * cost
    after = (n - skipped) * cost
    saving = before - after - fee
    payback = setup / saving if saving > 0 else None
    errors = skipped * (1 - chosen["accuracy"])
    with right:
        kpi_row([
            ("Без полной дегустации", f"{chosen['share']:.0%}", f"≈ {skipped:.0f} проб в месяц", ch.WHITE_C),
            ("Чистая экономия", f"{saving:,.0f} ₽".replace(",", " "), "в месяц, с учётом подписки",
             ch.BAND_C["высокое"] if saving > 0 else ch.NEG_C),
            ("Окупаемость", f"{payback:.1f} мес" if payback else "—", "внедрения", ch.GOLD),
            ("Риск ошибки", f"≈ {errors:.1f}", f"проб в месяц (точность {chosen['accuracy']:.0%})", ch.NEG_C),
        ])
        st.write("")
        c1, c2 = st.columns([1.4, 1], gap="medium")
        with c1, st.container(key="card_econ_curve"):
            st.markdown('<div class="section">Точность против доли проб без дегустации</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.triage_tradeoff(curve, chosen), config=PLOT_CFG, width="stretch")
        with c2, st.container(key="card_econ_cost"):
            st.markdown('<div class="section">Затраты в месяц</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.monthly_costs(before, after, fee), config=PLOT_CFG, width="stretch")
    st.write("")
    with st.container(key="card_econ_market"):
        st.markdown('<div class="section">Рынок и эффект для отрасли</div>', unsafe_allow_html=True)
        kpi_row([
            ("Производство вина в России, 2025", "54+ млн дал", "оценка Ассоциации виноделов", None),
            ("Малых виноделен (КФХ и ИП)", "111", "осень 2025; было 55 в 2021 году", ch.GOLD),
            ("За год, ваше предприятие", f"{saving * 12 - setup:,.0f} ₽".replace(",", " "),
             "чистый эффект первого года", ch.BAND_C["высокое"] if saving * 12 > setup else ch.NEG_C),
        ])
        st.caption("Источники: finance.mail.ru (Ассоциация виноделов, 2025), «Российская газета» (13.11.2025). "
                   "Кроме экономии на дегустациях: прогноз за секунды вместо ожидания комиссии и ранний сигнал "
                   "о возможном браке до розлива.")
