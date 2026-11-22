"""Загрузка датасета Wine Quality (UCI, id=186) и приведение пользовательских данных к единому виду."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
FILES = {"red": "winequality-red.csv", "white": "winequality-white.csv"}

# 11 физико-химических показателей из датасета (имена как в исходных CSV)
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
MODEL_FEATURES = FEATURES + ["is_red"]
TARGET = "quality"

RU_NAMES = {
    "fixed acidity": "Фиксированная кислотность, г/дм³",
    "volatile acidity": "Летучая кислотность, г/дм³",
    "citric acid": "Лимонная кислота, г/дм³",
    "residual sugar": "Остаточный сахар, г/дм³",
    "chlorides": "Хлориды, г/дм³",
    "free sulfur dioxide": "Свободный SO₂, мг/дм³",
    "total sulfur dioxide": "Общий SO₂, мг/дм³",
    "density": "Плотность, г/см³",
    "pH": "pH",
    "sulphates": "Сульфаты, г/дм³",
    "alcohol": "Спирт, % об.",
    "is_red": "Красное вино (1/0)",
    "quality": "Оценка качества",
}
RU_SHORT = {
    "fixed acidity": "Фикс. кислотность", "volatile acidity": "Летучая кислотность",
    "citric acid": "Лимонная кислота", "residual sugar": "Остаточный сахар",
    "chlorides": "Хлориды", "free sulfur dioxide": "Свободный SO₂",
    "total sulfur dioxide": "Общий SO₂", "density": "Плотность", "pH": "pH",
    "sulphates": "Сульфаты", "alcohol": "Спирт", "is_red": "Тип (красное)",
    "quality": "Качество",
}

# Синонимы колонок, чтобы принимать файлы лаборатории в разных написаниях
_ALIASES = {
    **{f: f for f in FEATURES},
    "fixed_acidity": "fixed acidity", "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid", "residual_sugar": "residual sugar",
    "free_sulfur_dioxide": "free sulfur dioxide", "total_sulfur_dioxide": "total sulfur dioxide",
    "фиксированная кислотность": "fixed acidity", "летучая кислотность": "volatile acidity",
    "лимонная кислота": "citric acid", "остаточный сахар": "residual sugar",
    "хлориды": "chlorides", "свободный so2": "free sulfur dioxide",
    "общий so2": "total sulfur dioxide", "плотность": "density",
    "сульфаты": "sulphates", "спирт": "alcohol", "алкоголь": "alcohol",
    "тип": "type", "type": "type", "color": "type", "is_red": "is_red",
}
_RED = {"red", "красное", "красный", "r", "1", "1.0", "true"}
_WHITE = {"white", "белое", "белый", "w", "0", "0.0", "false"}


def _norm(col: str) -> str:
    c = str(col).strip().lower().replace("_", " ")
    c = c.split(",")[0].split("(")[0].strip()  # отбрасываем единицы измерения
    c = c.replace("so₂", "so2")
    if c == "ph":
        return "pH"
    return _ALIASES.get(c, _ALIASES.get(c.replace(" ", "_"), c))


def download_dataset(data_dir: str | Path = "data") -> None:
    """Скачивает архив с UCI и распаковывает два CSV в data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю датасет: {UCI_ZIP_URL}")
    with urlopen(UCI_ZIP_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in FILES.values():
            member = next(m for m in zf.namelist() if m.endswith(name))
            (data_dir / name).write_bytes(zf.read(member))


def load_dataset(data_dir: str | Path = "data", download: bool = True) -> pd.DataFrame:
    """Возвращает объединённый датасет красных и белых вин с колонками
    FEATURES + type + is_red + quality."""
    data_dir = Path(data_dir)
    missing = [n for n in FILES.values() if not (data_dir / n).exists()]
    if missing:
        if not download:
            raise FileNotFoundError(
                f"Нет файлов {missing} в {data_dir}. Скачайте архив с "
                "https://archive.ics.uci.edu/dataset/186/wine+quality и распакуйте сюда."
            )
        download_dataset(data_dir)

    frames = []
    for wine_type, name in FILES.items():
        df = pd.read_csv(data_dir / name, sep=";")
        df.columns = [_norm(c) if c != TARGET else c for c in df.columns]
        df["type"] = wine_type
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["is_red"] = (df["type"] == "red").astype(int)
    return df[FEATURES + ["type", "is_red", TARGET]]


def read_user_file(file, default_type: str = "white") -> pd.DataFrame:
    """Читает файл с результатами лабораторных измерений (CSV с ; или , либо Excel)."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, sep=None, engine="python")
    return prepare_input(df, default_type=default_type)


def prepare_input(df: pd.DataFrame, default_type: str = "white") -> pd.DataFrame:
    """Приводит таблицу к признакам модели. Бросает ValueError с понятным текстом при ошибках."""
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError("В файле не найдены колонки: " + ", ".join(missing))

    for f in FEATURES:
        df[f] = pd.to_numeric(df[f].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    bad_rows = df[FEATURES].isna().any(axis=1)
    if bad_rows.any():
        idx = ", ".join(str(i + 1) for i in df.index[bad_rows][:10])
        raise ValueError(f"Пустые или нечисловые значения в строках: {idx}")

    if "is_red" not in df.columns:
        if "type" in df.columns:
            t = df["type"].astype(str).str.strip().str.lower()
            unknown = ~t.isin(_RED | _WHITE)
            if unknown.any():
                raise ValueError("Колонка «type» должна содержать red/white или красное/белое")
            df["is_red"] = t.isin(_RED).astype(int)
        else:
            df["is_red"] = int(default_type == "red")
    df["is_red"] = pd.to_numeric(df["is_red"], errors="coerce").fillna(0).astype(int)
    return df
