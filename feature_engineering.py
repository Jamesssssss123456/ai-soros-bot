
import numpy as np

def prepare_features(features: dict):
    return [
        features["oi_change_pct"],
        features["basis_percent_negative"],
        features["top_trader_account_ls_ratio"],
        features["top_trader_position_ls_ratio"]
    ]

def calculate_tp_sl_risk(features: dict):
    price = features.get("mark_price", 1)
    oi_change = features.get("oi_change_pct", 0)
    basis = features.get("basis_percent_negative", 0)
    volatility = oi_change * basis

    risk_level = max(0.01, min(0.05, volatility))
    tp = round(price * (1 + 3 * risk_level), 4)
    sl = round(price * (1 - risk_level), 4)
    rr = round((tp - price) / (price - sl), 2) if (price - sl) != 0 else 0

    return tp, sl, rr
