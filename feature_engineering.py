def prepare_features(data_dict):
    return [
        data_dict["oi_change_pct"],
        data_dict["basis_percent_negative"],
        data_dict["top_trader_account_ls_ratio"],
        data_dict["top_trader_position_ls_ratio"]
    ]