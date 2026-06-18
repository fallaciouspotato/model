from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_DATA = Path("UncleanedDataset.csv")
ARTIFACT_DIR = Path("artifacts")
CLEANED_DATA = ARTIFACT_DIR / "cleaned_traffic_events.csv"
MODEL_PATH = ARTIFACT_DIR / "astra_priority_model.json"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
CORRIDOR_RISK_PATH = ARTIFACT_DIR / "corridor_stress_ranking.csv"
JUNCTION_RISK_PATH = ARTIFACT_DIR / "junction_risk_ranking.csv"

TARGET = "priority"
POSITIVE_CLASS = "High"
NEGATIVE_CLASS = "Low"

MODEL_FEATURES = [
    "event_type",
    "event_cause",
    "requires_road_closure",
    "corridor",
    "zone",
    "junction",
    "police_station",
    "gba_identifier",
    "veh_type",
    "direction",
    "has_end_location",
    "start_hour",
    "start_dayofweek",
    "start_month",
    "is_weekend",
    "event_duration_bucket",
    "latitude_bin",
    "longitude_bin",
]

DROP_COLUMNS = [
    "map_file",
    "comment",
    "meta_data",
]

LEAKAGE_OR_ID_COLUMNS = [
    "id",
    "created_by_id",
    "last_modified_by_id",
    "assigned_to_police_id",
    "citizen_accident_id",
    "kgid",
    "closed_by_id",
    "resolved_by_id",
    "modified_datetime",
    "created_date",
    "closed_datetime",
    "resolved_datetime",
    "resolved_at_address",
    "resolved_at_latitude",
    "resolved_at_longitude",
    "status",
    "authenticated",
    "description",
    "veh_no",
    "cargo_material",
    "reason_breakdown",
    "route_path",
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    text = str(value).strip()
    if not text:
        return "__missing__"
    return " ".join(text.split())


def normalize_category(value: Any) -> str:
    return normalize_text(value).lower().replace("-", "_")


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def make_duration_bucket(hours: float) -> str:
    if pd.isna(hours):
        return "unknown"
    if hours <= 1:
        return "0_1h"
    if hours <= 3:
        return "1_3h"
    if hours <= 6:
        return "3_6h"
    if hours <= 12:
        return "6_12h"
    if hours <= 24:
        return "12_24h"
    return "24h_plus"


def impact_category(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 80:
        return "High"
    return "Critical"


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for column in DROP_COLUMNS:
        if column in df.columns:
            df = df.drop(columns=column)

    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = df[column].map(normalize_text)
        df.loc[df[column] == "__missing__", column] = pd.NA

    if "event_cause" in df.columns:
        df["event_cause"] = df["event_cause"].map(normalize_category)
        df["event_cause"] = df["event_cause"].replace({"debris": "debris"})

    for column in ["event_type", "corridor", "zone", "junction", "police_station", "gba_identifier", "veh_type", "direction"]:
        if column in df.columns:
            df[column] = df[column].map(normalize_text)

    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = df["requires_road_closure"].astype("boolean").fillna(False).astype(bool)

    start_dt = parse_datetime(df["start_datetime"])
    end_dt = parse_datetime(df["end_datetime"]) if "end_datetime" in df.columns else pd.Series(pd.NaT, index=df.index)

    df["start_hour"] = start_dt.dt.hour.fillna(-1).astype(int)
    df["start_dayofweek"] = start_dt.dt.dayofweek.fillna(-1).astype(int)
    df["start_month"] = start_dt.dt.month.fillna(-1).astype(int)
    df["is_weekend"] = df["start_dayofweek"].isin([5, 6]).astype(int)

    duration_hours = (end_dt - start_dt).dt.total_seconds() / 3600
    duration_hours = duration_hours.where(duration_hours.ge(0))
    df["event_duration_hours"] = duration_hours.fillna(1.0).clip(lower=0, upper=72)
    df["event_duration_bucket"] = df["event_duration_hours"].map(make_duration_bucket)

    for column in ["endlatitude", "endlongitude"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[df[column].eq(0), column] = np.nan

    df["has_end_location"] = df[["endlatitude", "endlongitude"]].notna().all(axis=1).astype(int)
    df["latitude_bin"] = pd.to_numeric(df["latitude"], errors="coerce").round(2).astype("string").fillna("__missing__")
    df["longitude_bin"] = pd.to_numeric(df["longitude"], errors="coerce").round(2).astype("string").fillna("__missing__")

    for column in MODEL_FEATURES:
        if column not in df.columns:
            df[column] = "__missing__"
        df[column] = df[column].map(normalize_text)

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map(normalize_text)
        df = df[df[TARGET].isin([POSITIVE_CLASS, NEGATIVE_CLASS])].copy()

    return df


def stratified_split(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, group in df.groupby(target):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        test_parts.append(df.loc[test_idx])
        train_parts.append(df.loc[train_idx])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test


def train_naive_bayes(train: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    classes = [NEGATIVE_CLASS, POSITIVE_CLASS]
    alpha = 1.0
    model: dict[str, Any] = {
        "model_type": "categorical_naive_bayes",
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "alpha": alpha,
        "features": features,
        "class_counts": {},
        "class_log_prior": {},
        "feature_log_probs": {},
        "feature_vocab": {},
    }

    total_rows = len(train)
    for cls in classes:
        class_count = int((train[target] == cls).sum())
        model["class_counts"][cls] = class_count
        model["class_log_prior"][cls] = math.log((class_count + alpha) / (total_rows + alpha * len(classes)))

    for feature in features:
        values = sorted(train[feature].map(normalize_text).unique().tolist())
        model["feature_vocab"][feature] = values
        model["feature_log_probs"][feature] = {}
        vocab_size = len(values) + 1
        for cls in classes:
            subset = train.loc[train[target] == cls, feature].map(normalize_text)
            counts = subset.value_counts().to_dict()
            denom = model["class_counts"][cls] + alpha * vocab_size
            probs = {value: math.log((counts.get(value, 0) + alpha) / denom) for value in values}
            probs["__unknown__"] = math.log(alpha / denom)
            model["feature_log_probs"][feature][cls] = probs

    return model


def predict_proba_high(model: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    probs = []
    classes = [model["negative_class"], model["positive_class"]]

    for _, row in rows.iterrows():
        log_scores = {}
        for cls in classes:
            score = model["class_log_prior"][cls]
            for feature in model["features"]:
                value = normalize_text(row.get(feature, "__missing__"))
                feature_probs = model["feature_log_probs"][feature][cls]
                score += feature_probs.get(value, feature_probs["__unknown__"])
            log_scores[cls] = score

        max_log = max(log_scores.values())
        exp_neg = math.exp(log_scores[model["negative_class"]] - max_log)
        exp_pos = math.exp(log_scores[model["positive_class"]] - max_log)
        probs.append(exp_pos / (exp_neg + exp_pos))

    return np.array(probs)


def roc_auc_score_manual(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(y_true_labels: pd.Series, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = (y_true_labels.to_numpy() == POSITIVE_CLASS).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "rows_evaluated": int(len(y_true)),
        "accuracy": round(float(accuracy), 4),
        "precision_high": round(float(precision), 4),
        "recall_high": round(float(recall), 4),
        "f1_high": round(float(f1), 4),
        "roc_auc": round(roc_auc_score_manual(y_true, y_prob), 4),
        "confusion_matrix": {
            "true_low_pred_low": tn,
            "true_low_pred_high": fp,
            "true_high_pred_low": fn,
            "true_high_pred_high": tp,
        },
    }


def add_predictions(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    output = df.copy()
    prob_high = predict_proba_high(model, output)
    output["congestion_risk_score"] = np.round(prob_high * 100, 2)
    output["event_severity_score"] = output["congestion_risk_score"]
    output["predicted_priority"] = np.where(prob_high >= 0.5, POSITIVE_CLASS, NEGATIVE_CLASS)
    output["impact_category"] = output["congestion_risk_score"].map(impact_category)
    return output


def build_rankings(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corridor = (
        scored.groupby("corridor", dropna=False)
        .agg(
            event_count=("corridor", "size"),
            avg_risk_score=("congestion_risk_score", "mean"),
            high_priority_rate=("predicted_priority", lambda s: float((s == POSITIVE_CLASS).mean())),
        )
        .reset_index()
    )
    corridor["corridor_stress_index"] = (
        0.75 * corridor["avg_risk_score"] + 0.25 * np.log1p(corridor["event_count"]) / np.log1p(corridor["event_count"].max()) * 100
    ).round(2)
    corridor = corridor.sort_values("corridor_stress_index", ascending=False)

    junction_source = scored[scored["junction"] != "__missing__"].copy()
    junction = (
        junction_source.groupby(["junction", "corridor"], dropna=False)
        .agg(
            event_count=("junction", "size"),
            avg_risk_score=("congestion_risk_score", "mean"),
            high_priority_rate=("predicted_priority", lambda s: float((s == POSITIVE_CLASS).mean())),
        )
        .reset_index()
    )
    junction["junction_risk_score"] = (
        0.8 * junction["avg_risk_score"] + 0.2 * np.log1p(junction["event_count"]) / np.log1p(junction["event_count"].max()) * 100
    ).round(2)
    junction = junction.sort_values("junction_risk_score", ascending=False)
    junction["priority_rank"] = np.arange(1, len(junction) + 1)

    return corridor, junction


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Could not find {RAW_DATA}")

    ARTIFACT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(RAW_DATA)
    cleaned = clean_data(raw)
    model_frame = cleaned.drop(columns=[c for c in LEAKAGE_OR_ID_COLUMNS if c in cleaned.columns], errors="ignore")

    train, test = stratified_split(model_frame, TARGET)
    model = train_naive_bayes(train, MODEL_FEATURES, TARGET)
    test_prob = predict_proba_high(model, test)
    metrics = evaluate(test[TARGET], test_prob)
    metrics.update(
        {
            "target": TARGET,
            "positive_class": POSITIVE_CLASS,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "features": MODEL_FEATURES,
        }
    )

    scored = add_predictions(cleaned, model)
    corridor_ranking, junction_ranking = build_rankings(scored)

    scored.to_csv(CLEANED_DATA, index=False)
    corridor_ranking.to_csv(CORRIDOR_RISK_PATH, index=False)
    junction_ranking.to_csv(JUNCTION_RISK_PATH, index=False)

    with MODEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("ASTRA model training complete")
    print(f"Cleaned rows: {len(scored)}")
    print(f"Model metrics: {json.dumps(metrics, indent=2)}")
    print(f"Wrote {CLEANED_DATA}")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {CORRIDOR_RISK_PATH}")
    print(f"Wrote {JUNCTION_RISK_PATH}")


if __name__ == "__main__":
    main()
