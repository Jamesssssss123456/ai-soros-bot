
def prepare_features(data):
    return [
        data.get("oi_change_pct", 0),
        data.get("basis_percent_negative", 0),
        data.get("top_trader_account_ls_ratio", 0),
        data.get("top_trader_position_ls_ratio", 0)
    ]

def calculate_tp_sl_risk(data):
    mark_price = data.get("mark_price", 1)
    tp = round(mark_price * 1.05, 4)
    sl = round(mark_price * 0.97, 4)
    rr = round((tp - mark_price) / (mark_price - sl), 2) if mark_price != sl else 1.0
    return tp, sl, rr
