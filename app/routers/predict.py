from fastapi import APIRouter, Depends, Query
from app.model import predict_flood_risk, FEATURES, is_model_loaded, get_registry_status
from app.schemas import NodePredictRequest, NodePredictResponse
from app.security import require_ai_service_key
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import date

router = APIRouter(dependencies=[Depends(require_ai_service_key)])

MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
RISK_LABELS = ["Normal", "Alert", "Warning", "Critical"]


def _map_proba_to_level(proba: float) -> int:
    if proba < 30:
        return 0
    if proba < 50:
        return 1
    if proba < 75:
        return 2
    return 3


def _seasonal_features(n: int, start_doy: int = 1, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Sarawak seasonal weather features for n time steps.

    Models the bimodal monsoon pattern:
    - Northeast Monsoon (Oct-Feb): high rainfall, elevated flood risk
    - Southwest Monsoon (May-Sep): moderate rainfall
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        doy = ((start_doy + i - 1) % 365) + 1
        # Northeast Monsoon peaks around day 30 (Jan 30)
        monsoon = 0.4 + 0.6 * (np.sin(2 * np.pi * (doy - 30) / 365) ** 2)
        base_rain = float(monsoon * 32 + rng.exponential(6))
        rows.append({
            'rain_1day': max(0.0, base_rain + rng.normal(0, 5)),
            'elevation': 15.0,
            'slope': 3.0,
            'u10': float(rng.normal(0, 2.5)),
            'v10': float(rng.normal(0, 2.5)),
            't2m': float(27.0 - monsoon * 1.5 + rng.normal(0, 0.5)),
            'sp': float(101325.0 + rng.normal(0, 300)),
            'tp': max(0.0, base_rain * 0.8 + float(rng.normal(0, 3))),
            'ro': max(0.0, base_rain * 0.3 + float(rng.exponential(2))),
            'swvl1': max(0.0, min(0.5, monsoon * 0.4 + float(rng.normal(0, 0.04)))),
            'rain_3day': max(0.0, base_rain * 2.5 + float(rng.normal(0, 9))),
            'rain_5day': max(0.0, base_rain * 4.0 + float(rng.normal(0, 14))),
            'rain_7day': max(0.0, base_rain * 5.5 + float(rng.normal(0, 18))),
            'rain_avg': max(0.0, base_rain * 0.9 + float(rng.normal(0, 2))),
            'wind_speed': abs(float(rng.normal(0, 3))),
            'storm_intensity': max(0.0, min(1.0, monsoon * 0.65 + float(rng.exponential(0.07)))),
            'slope_runoff_potential': float(rng.uniform(0.2, 0.8)),
        })
    return pd.DataFrame(rows)


@router.get("/predict/daily", summary="365-day flood risk predictions")
async def predict_daily(year: int = Query(default=2026, ge=2020, le=2100)) -> dict:
    np.random.seed(year % 1000)
    df = _seasonal_features(365, start_doy=1, seed=year % 1000)
    preds = predict_flood_risk(df)

    daily_data: dict = OrderedDict()
    hourly_data: dict = OrderedDict()
    averages: dict = {}
    idx = 0

    for i, month in enumerate(MONTHS):
        n = DAYS_IN_MONTH[i]
        month_preds = preds[idx: idx + n]
        probas = [p['probability'] for p in month_preds]
        levels = [p['level'] for p in month_preds]

        daily_data[month] = levels
        averages[month] = round(sum(probas) / len(probas), 1) if probas else 0.0

        hourly_data[month] = {}
        rng = np.random.RandomState(year + i)
        for d in range(n):
            base = probas[d] if d < len(probas) else averages[month]
            hourly_data[month][f"{month} {d + 1}"] = [
                _map_proba_to_level(max(0.0, min(100.0, base + float(rng.uniform(-12, 12)))))
                for _ in range(24)
            ]
        idx += n

    return {
        "year": year,
        "scale": "daily",
        "daily_data": daily_data,
        "hourly_data": hourly_data,
        "averages": averages,
        "model_loaded": is_model_loaded(),
        "model_version": "xgboost-v1.0",
    }


@router.get("/predict/weekly", summary="Quarterly (Q1-Q4) flood risk predictions")
async def predict_weekly(year: int = Query(default=2026, ge=2020, le=2100)) -> dict:
    np.random.seed(year % 1000 + 1)
    df = _seasonal_features(365, seed=year % 1000 + 1)
    preds = predict_flood_risk(df)
    proba_list = [p['probability'] for p in preds]

    weekly_avgs = [
        round(sum(proba_list[i: i + 7]) / max(1, len(proba_list[i: i + 7])), 1)
        for i in range(0, 364, 7)
    ]

    return {
        "year": year,
        "scale": "weekly",
        "data": {
            "Q1 (Jan-Mar)": [_map_proba_to_level(w) for w in weekly_avgs[0:13]],
            "Q2 (Apr-Jun)": [_map_proba_to_level(w) for w in weekly_avgs[13:26]],
            "Q3 (Jul-Sep)": [_map_proba_to_level(w) for w in weekly_avgs[26:39]],
            "Q4 (Oct-Dec)": [_map_proba_to_level(w) for w in weekly_avgs[39:52]],
        },
        "model_loaded": is_model_loaded(),
        "model_version": "xgboost-v1.0",
    }


@router.get("/predict/monthly", summary="Monthly average flood risk predictions")
async def predict_monthly(year: int = Query(default=2026, ge=2020, le=2100)) -> dict:
    np.random.seed(year % 1000 + 2)
    df = _seasonal_features(365, seed=year % 1000 + 2)
    preds = predict_flood_risk(df)

    monthly = []
    idx = 0
    for i, month in enumerate(MONTHS):
        n = DAYS_IN_MONTH[i]
        month_preds = preds[idx: idx + n]
        avg_proba = sum(p['probability'] for p in month_preds) / max(1, len(month_preds))
        monthly.append({
            "month": month,
            "month_index": i + 1,
            "level": _map_proba_to_level(avg_proba),
            "avg_probability": round(avg_proba, 1),
            "risk_label": RISK_LABELS[_map_proba_to_level(avg_proba)],
        })
        idx += n

    return {
        "year": year,
        "scale": "monthly",
        "data": monthly,
        "model_loaded": is_model_loaded(),
        "model_version": "xgboost-v1.0",
    }


@router.get("/predict/hourly", summary="24-hour flood risk predictions for a specific date")
async def predict_hourly(
    date_str: str = Query(
        alias="date",
        default="2026-04-28",
        description="ISO date e.g. 2026-04-28",
    ),
) -> dict:
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        d = date.today()

    doy = d.timetuple().tm_yday
    seed = doy * 100 + d.year % 100
    df = _seasonal_features(1, start_doy=doy, seed=seed)
    base_proba = predict_flood_risk(df)[0]['probability']

    rng = np.random.RandomState(seed)
    hourly = []
    for h in range(24):
        p = max(0.0, min(100.0, base_proba + float(rng.uniform(-12, 12))))
        hourly.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "level": _map_proba_to_level(p),
            "probability": round(p, 1),
            "risk_label": RISK_LABELS[_map_proba_to_level(p)],
        })

    return {
        "date": date_str,
        "scale": "hourly",
        "data": hourly,
        "model_loaded": is_model_loaded(),
        "model_version": "xgboost-v1.0",
    }


@router.post("/predict/node", response_model=NodePredictResponse, summary="Predict risk for a specific sensor node")
async def predict_node(body: NodePredictRequest) -> NodePredictResponse:
    water_level = body.water_level or 0
    rain_boost = max(0.0, (water_level - 1) * 15.0)

    row = {
        'rain_1day': (body.rain_1day or 10.0) + rain_boost,
        'elevation': body.elevation or 15.0,
        'slope': body.slope or 3.0,
        'u10': 0.0,
        'v10': 0.0,
        't2m': 27.0,
        'sp': 101325.0,
        'tp': (body.rain_1day or 10.0) * 0.8,
        'ro': (body.rain_1day or 10.0) * 0.3,
        'swvl1': 0.3,
        'rain_3day': (body.rain_3day or 25.0) + rain_boost * 2.5,
        'rain_5day': (body.rain_5day or 40.0) + rain_boost * 4,
        'rain_7day': (body.rain_7day or 55.0) + rain_boost * 5.5,
        'rain_avg': (body.rain_avg or 9.0) + rain_boost * 0.5,
        'wind_speed': body.wind_speed or 3.0,
        'storm_intensity': body.storm_intensity or 0.1,
        'slope_runoff_potential': 0.5,
    }

    df = pd.DataFrame([row])
    result = predict_flood_risk(df)[0]

    return NodePredictResponse(
        node_id=body.node_id,
        predicted_level=result['level'],
        probability=result['probability'],
        risk_label=RISK_LABELS[result['level']],
        model_used="xgboost" if is_model_loaded() else "rule-based-fallback",
    )


@router.get("/model/info", summary="Model registry and inference metadata")
async def model_info() -> dict:
    registry = get_registry_status()
    loaded_names = [k for k, v in registry.items() if v["loaded"]]
    return {
        "inference_mode": "weighted-ensemble" if len(loaded_names) > 1 else ("single-model" if loaded_names else "rule-based-fallback"),
        "loaded_models": loaded_names,
        "model_registry": registry,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "output_levels": {str(i): RISK_LABELS[i] for i in range(4)},
        "probability_thresholds": {
            "Normal (0)": "< 30 %",
            "Alert (1)":  "30 – 50 %",
            "Warning (2)": "50 – 75 %",
            "Critical (3)": "≥ 75 %",
        },
        "training_region": "Sarawak / Sabah, Malaysia",
        "model_version": "multi-model-v2.0",
        "sources": [
            "FYP-RainfallView (BrianChristopherJohan) — XGBoost v2, LightGBM, CatBoost",
            "flood-ai-prediction scripts/train.py — XGBoost retrained on 15k Sarawak synthetic samples",
        ],
    }
