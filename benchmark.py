#!/usr/bin/env python3
"""
Benchmark Suite: Automotive Valuation & Hedonic Price Estimator
Evaluates Ridge Regression (L2), Lasso Regression (L1), and Gradient Boosting Regressor
using 5-Fold Cross-Validation on secondary-market automotive depreciation features.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_automotive_hedonic_dataset(n_samples: int = 850, random_state: int = 42):
    """
    Synthesizes an automotive secondary-market dataset following exponential age/mileage
    depreciation and hedonic attribute valuation (Showroom Price, Vehicle Age,
    Kilometers Driven, Fuel Type, Transmission, Seller Type, Previous Owners).
    """
    rng = np.random.default_rng(random_state)

    present_price = rng.lognormal(mean=2.05, sigma=0.62, size=n_samples)  # in Lakhs / $10k units
    vehicle_age = rng.uniform(0.5, 13.0, size=n_samples)
    kms_driven = vehicle_age * rng.uniform(6500, 16500, size=n_samples) + rng.normal(0, 2200, size=n_samples)
    kms_driven = np.clip(kms_driven, 1200, 220000)
    is_diesel = rng.binomial(1, 0.32, size=n_samples)
    is_automatic = rng.binomial(1, 0.26, size=n_samples)
    is_dealer = rng.binomial(1, 0.68, size=n_samples)
    owners = rng.choice([0, 1, 2], size=n_samples, p=[0.82, 0.15, 0.03])

    # Non-linear hedonic depreciation curve + interaction effects
    depreciation_factor = np.exp(-0.105 * vehicle_age - 0.0000024 * kms_driven)
    premium_multiplier = (
        1.0
        + 0.085 * is_diesel
        + 0.070 * is_automatic
        + 0.055 * is_dealer
        - 0.065 * owners
    )
    noise = rng.normal(0, 0.42, size=n_samples)
    selling_price = np.clip(present_price * depreciation_factor * premium_multiplier + noise, 0.35, None)

    X = np.column_stack([
        present_price,
        vehicle_age,
        kms_driven,
        is_diesel,
        is_automatic,
        is_dealer,
        owners,
    ])
    feature_names = [
        "Present_Showroom_Price",
        "Vehicle_Age_Years",
        "Kilometers_Driven",
        "Fuel_Diesel_Indicator",
        "Transmission_Auto_Indicator",
        "Seller_Dealer_Indicator",
        "Previous_Owners",
    ]
    return X, selling_price, feature_names


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for automotive hedonic regression."""
    width, height = 860, 420
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Automotive Valuation &amp; Hedonic Price Estimator — 5-Fold CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Secondary Market Cohort (N = 850 Vehicles, d = 7 Hedonic Attributes) | R² &amp; Error Metrics</text>',
        '  <rect x="32" y="84" width="12" height="12" rx="3" fill="#38bdf8"/>',
        '  <text x="50" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">R² Score (Higher is Better)</text>',
        '  <rect x="245" y="84" width="12" height="12" rx="3" fill="#34d399"/>',
        '  <text x="263" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">1 - Normalized RMSE</text>',
        '  <rect x="440" y="84" width="12" height="12" rx="3" fill="#f59e0b"/>',
        '  <text x="458" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">MAE (Price Units — Lower is Better)</text>',
    ]

    chart_left = 255
    chart_right = 795
    chart_width = chart_right - chart_left
    top_y = 124
    group_height = 90

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 12}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        # Bar 1: R2
        r2_w = max(2.0, entry["r2"] * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy}" width="{r2_w:.1f}" height="16" rx="3" fill="#38bdf8"/>')
        svg.append(
            f'  <text x="{chart_left + r2_w + 8:.1f}" y="{gy + 12}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">R² = {entry["r2"]:.4f}</text>'
        )
        # Bar 2: 1 - NRMSE
        nrmse_score = max(0.0, 1.0 - entry["rmse"] / 5.0)
        nrmse_w = max(2.0, nrmse_score * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 21}" width="{nrmse_w:.1f}" height="16" rx="3" fill="#34d399"/>')
        svg.append(
            f'  <text x="{chart_left + nrmse_w + 8:.1f}" y="{gy + 33}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">RMSE = {entry["rmse"]:.4f}</text>'
        )
        # Bar 3: MAE (scaled to chart)
        mae_w = max(2.0, min(1.0, entry["mae"] / 2.0) * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 42}" width="{mae_w:.1f}" height="16" rx="3" fill="#f59e0b"/>')
        svg.append(
            f'  <text x="{chart_left + mae_w + 8:.1f}" y="{gy + 54}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">MAE = {entry["mae"]:.4f}</text>'
        )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X, y, feature_names = generate_automotive_hedonic_dataset()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=220, learning_rate=0.06, max_depth=4, random_state=42
        ),
        "Ridge Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=2.0, random_state=42)),
        ]),
        "Lasso Regression (L1)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.03, random_state=42)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X[val_idx])

        rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
        mae = float(mean_absolute_error(y, oof_preds))
        mape = float(np.mean(np.abs((y - oof_preds) / y)) * 100.0)
        r2 = float(r2_score(y, oof_preds))

        benchmark_rows.append({
            "model": name,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape_pct": round(mape, 2),
        })

    benchmark_rows.sort(key=lambda r: r["r2"], reverse=True)

    payload = {
        "project": "Automotive Valuation & Hedonic Price Estimator",
        "dataset": "Secondary Automotive Market Depreciation Cohort (N=850, d=7)",
        "validation_scheme": "5-Fold Cross-Validation",
        "features": feature_names,
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
