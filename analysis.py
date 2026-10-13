from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Location_ID",
    "Location_Name",
    "Floor",
    "X_m",
    "Y_m",
    "Time_Period",
    "RSSI_dBm",
]

WEAK_THRESHOLD_DEFAULT = -67.0


class DataValidationError(ValueError):
    """Raised when the survey data cannot safely enter the analysis pipeline."""


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise DataValidationError("The dataset is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))

    seen = set()
    for r in rows:
        key = (str(r.get("Location_ID", "")), str(r.get("Time_Period", "")))
        if key in seen:
            raise DataValidationError(
                f"Duplicate location/time record: {key[0]} / {key[1]}"
            )
        seen.add(key)

        for key_num in ("X_m", "Y_m", "RSSI_dBm"):
            try:
                value = float(r[key_num])
            except (TypeError, ValueError, KeyError):
                raise DataValidationError(
                    f"Invalid numeric value in {key_num} for {r.get('Location_ID', '?')}."
                )
            if not np.isfinite(value):
                raise DataValidationError(
                    f"Non-finite numeric value in {key_num} for {r.get('Location_ID', '?')}."
                )

    locations = list(OrderedDict.fromkeys(str(r["Location_ID"]) for r in rows))
    periods = list(OrderedDict.fromkeys(str(r["Time_Period"]) for r in rows))

    if len(locations) < 4:
        raise DataValidationError("At least four survey locations are required.")
    if len(periods) < 2:
        raise DataValidationError("At least two time periods are required.")

    expected = {(loc, period) for loc in locations for period in periods}
    if seen != expected:
        missing_pairs = sorted(expected - seen)
        preview = ", ".join(f"{a}/{b}" for a, b in missing_pairs[:5])
        suffix = "..." if len(missing_pairs) > 5 else ""
        raise DataValidationError(
            f"Survey is incomplete; every location needs every period. Missing: {preview}{suffix}"
        )


def _records_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    for c in ("X_m", "Y_m", "RSSI_dBm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_matrix(df: pd.DataFrame, locations: list[str], periods: list[str]) -> np.ndarray:
    pivot = (
        df.pivot(index="Location_ID", columns="Time_Period", values="RSSI_dBm")
        .reindex(index=locations, columns=periods)
    )
    if pivot.isna().any().any():
        raise DataValidationError("Matrix contains missing location/time measurements.")
    return pivot.to_numpy(dtype=float)


def normalize_location_vectors(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    # A zero-variance time vector has no unique direction; keep it finite.
    norms[norms < 1e-12] = 1.0
    return centered / norms


def similarity_analysis(matrix: np.ndarray, locations: list[str]) -> dict[str, Any]:
    vectors = normalize_location_vectors(matrix)
    cosine = vectors @ vectors.T
    distance = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)

    pairs = []
    n = len(locations)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "locA": locations[i],
                    "locB": locations[j],
                    "cos": float(cosine[i, j]),
                    "dist": float(distance[i, j]),
                }
            )

    pairs.sort(key=lambda p: (-p["cos"], p["dist"], p["locA"], p["locB"]))
    return {
        "normVecs": vectors.tolist(),
        "cosMatrix": cosine.tolist(),
        "distMatrix": distance.tolist(),
        "pairs": pairs,
    }


def fit_least_squares_plane(df: pd.DataFrame) -> list[float]:
    X = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            df["X_m"].to_numpy(dtype=float),
            df["Y_m"].to_numpy(dtype=float),
        ]
    )
    y = df["RSSI_dBm"].to_numpy(dtype=float)

    # Explicit least-squares solution. np.linalg.lstsq handles the
    # numerical solve robustly while remaining a direct Linear Algebra method.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return [float(v) for v in beta]


def predict_rssi(beta: list[float], x: float, y: float) -> float:
    return float(beta[0] + beta[1] * x + beta[2] * y)


def build_spatial_models(
    df: pd.DataFrame, locations: list[str], periods: list[str]
) -> dict[str, list[float]]:
    models: dict[str, list[float]] = {}
    for period in periods:
        subset = df[df["Time_Period"].astype(str) == period]
        models[period] = fit_least_squares_plane(subset)

    avg = (
        df.groupby("Location_ID", sort=False)
        .agg(
            Location_Name=("Location_Name", "first"),
            Floor=("Floor", "first"),
            X_m=("X_m", "first"),
            Y_m=("Y_m", "first"),
            RSSI_dBm=("RSSI_dBm", "mean"),
        )
        .reindex(locations)
        .reset_index()
    )
    models["Average"] = fit_least_squares_plane(avg)
    return models


def held_out_validation(
    df: pd.DataFrame, locations: list[str], periods: list[str]
) -> dict[str, Any]:
    n = len(locations)
    indices = [0, n // 3, (2 * n) // 3, n - 1]
    hold_locs = list(OrderedDict.fromkeys(locations[i] for i in indices))

    records = []
    y_true: list[float] = []

    for period in periods:
        train = df[
            (df["Time_Period"].astype(str) == period)
            & (~df["Location_ID"].astype(str).isin(hold_locs))
        ]
        test = df[
            (df["Time_Period"].astype(str) == period)
            & (df["Location_ID"].astype(str).isin(hold_locs))
        ]
        beta = fit_least_squares_plane(train)
        for _, row in test.iterrows():
            actual = float(row["RSSI_dBm"])
            predicted = predict_rssi(beta, float(row["X_m"]), float(row["Y_m"]))
            residual = actual - predicted
            records.append(
                {
                    "Location_ID": str(row["Location_ID"]),
                    "Time_Period": str(period),
                    "Actual_RSSI": actual,
                    "Predicted_RSSI": predicted,
                    "Absolute_Error": abs(residual),
                    "Residual": residual,
                }
            )
            y_true.append(actual)

    errors = np.asarray([r["Absolute_Error"] for r in records], dtype=float)
    residuals = np.asarray([r["Residual"] for r in records], dtype=float)
    mae = float(np.mean(errors)) if len(errors) else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    yt = np.asarray(y_true, dtype=float)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) if len(yt) else 0.0
    ss_res = float(np.sum(residuals**2)) if len(residuals) else 0.0
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return {
        "records": records,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "heldOutLocations": hold_locs,
    }


def weak_zone_analysis(
    df: pd.DataFrame, threshold: float = WEAK_THRESHOLD_DEFAULT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = (
        df.groupby("Location_ID", sort=False)
        .agg(
            Location_Name=("Location_Name", "first"),
            Floor=("Floor", "first"),
            X_m=("X_m", "first"),
            Y_m=("Y_m", "first"),
            avg_rssi=("RSSI_dBm", "mean"),
            min_rssi=("RSSI_dBm", "min"),
            max_rssi=("RSSI_dBm", "max"),
        )
        .reset_index()
    )

    counts = (
        df.assign(weak=df["RSSI_dBm"] <= threshold)
        .groupby("Location_ID", sort=False)["weak"]
        .sum()
        .rename("weak_count")
        .reset_index()
    )
    stats = stats.merge(counts, on="Location_ID", how="left")
    stats["reading_count"] = (
        df.groupby("Location_ID", sort=False)["RSSI_dBm"].size().reindex(stats["Location_ID"]).to_numpy()
    )
    stats["isWeak"] = (
        (stats["avg_rssi"] <= threshold)
        & (stats["weak_count"] >= np.ceil(stats["reading_count"] / 2.0))
    )
    stats["isStrong"] = stats["avg_rssi"] >= -55.0

    def row_to_dict(r):
        values = (
            df[df["Location_ID"].astype(str) == str(r["Location_ID"])]["RSSI_dBm"]
            .astype(float)
            .tolist()
        )
        return {
            "Location_ID": str(r["Location_ID"]),
            "Location_Name": str(r["Location_Name"]),
            "Floor": int(r["Floor"]) if pd.notna(r["Floor"]) else 0,
            "X_m": float(r["X_m"]),
            "Y_m": float(r["Y_m"]),
            "values": values,
            "avg_rssi": float(r["avg_rssi"]),
            "min_rssi": float(r["min_rssi"]),
            "max_rssi": float(r["max_rssi"]),
            "weak_count": int(r["weak_count"]),
            "isWeak": bool(r["isWeak"]),
            "isStrong": bool(r["isStrong"]),
        }

    all_stats = [row_to_dict(r) for _, r in stats.iterrows()]
    weak = sorted([x for x in all_stats if x["isWeak"]], key=lambda x: x["avg_rssi"])
    strong = sorted([x for x in all_stats if x["isStrong"]], key=lambda x: -x["avg_rssi"])
    return weak, strong


def router_reference(
    df: pd.DataFrame, beta: list[float], threshold: float
) -> dict[str, float]:
    xmin, xmax = float(df["X_m"].min()), float(df["X_m"].max())
    ymin, ymax = float(df["Y_m"].min()), float(df["Y_m"].max())
    grid_res = 30
    best_coord = (xmin, ymin)
    best_val = -np.inf
    adequate = 0
    total = grid_res * grid_res

    for y in np.linspace(ymin, ymax, grid_res):
        for x in np.linspace(xmin, xmax, grid_res):
            value = predict_rssi(beta, float(x), float(y))
            if value >= threshold:
                adequate += 1
            if value > best_val:
                best_val = value
                best_coord = (float(x), float(y))

    return {
        "x": round(best_coord[0], 2),
        "y": round(best_coord[1], 2),
        "coveragePct": float(adequate / total * 100.0),
        "peakSignal": float(best_val),
    }


def eigen_analysis(matrix: np.ndarray) -> dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    n = centered.shape[0]
    covariance = centered.T @ centered / max(n - 1, 1)

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order].T
    values = np.maximum(values, 0.0)
    total = float(values.sum()) or 1.0
    explained = values / total

    return {
        "values": values.tolist(),
        "vectors": vectors.tolist(),
        "explained": explained.tolist(),
    }


def analyse_rows(
    rows: list[dict[str, Any]], threshold: float = WEAK_THRESHOLD_DEFAULT
) -> dict[str, Any]:
    validate_rows(rows)
    df = _records_to_frame(rows)

    locations = list(OrderedDict.fromkeys(df["Location_ID"].astype(str).tolist()))
    periods = list(OrderedDict.fromkeys(df["Time_Period"].astype(str).tolist()))
    matrix = build_matrix(df, locations, periods)

    similarity = similarity_analysis(matrix, locations)
    models = build_spatial_models(df, locations, periods)
    validation = held_out_validation(df, locations, periods)
    weak, strong = weak_zone_analysis(df, threshold)
    reference = router_reference(df, models["Average"], threshold)
    eigen = eigen_analysis(matrix)

    return {
        "locations": locations,
        "periods": periods,
        "matrix": matrix.tolist(),
        "similarity": similarity,
        "models": models,
        "validation": validation,
        "weakZones": weak,
        "strongZones": strong,
        "routerPlacement": reference,
        "eigen": eigen,
        "threshold": float(threshold),
    }
