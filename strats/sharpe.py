from data.candles_eodhd import download_symbol_candles


def sharpe_ratio_example():
    # example of calculating sharpe ratio for trivial strategy
    # sharpe ratio is a metric that shows how some strategy compares to "risk-free" rate benchmark (usually U.S. Treasury bill)
    # generally sharpe ratio calculated as avg(excessive_return_t) / (std(excessive_return_t)),
    # where excessive_return_t is (strategy_return_t - risk_free_return_t)

    from_iso = "2026-01-01"
    to_iso = "2026-10-24"
    to_long_df = download_symbol_candles("AAPL.US", from_iso, to_iso,
                                         period='d', to_cache=True)

    # assumed annual risk-free rate, 252 days in trading year
    T = 252
    daily_risk_free_rate = 0.04 / T

    daily_ret_raw_a = ((to_long_df["close"].shift(-1) - to_long_df["close"]) / to_long_df["close"]).dropna()
    # minus risk-free rate
    daily_ret_a = daily_ret_raw_a - daily_risk_free_rate

    # to calculate annualized sharpe ratio, we multiply sharpe by additional sqrt(N_t),
    # where N_t is the amount of periods 't' in 'T'
    daily_sharpe_ratio_a = daily_ret_a.mean() / daily_ret_a.std()
    annualized_sharpe_ratio_a = daily_sharpe_ratio_a * (T ** 0.5)
    print(f"long AAPL.US annual sharpe: {annualized_sharpe_ratio_a:.4f}")


    # let's add another stock in our strat, but instead of long we will short it with the same amount of money
    # btw, trading calendar should align
    to_short_df = download_symbol_candles("QQQ.US", from_iso, to_iso,
                                          period='d', to_cache=True)

    daily_ret_raw_b = ((to_short_df["close"].shift(-1) - to_short_df["close"]) / to_short_df["close"]).dropna()
    daily_ret_b = daily_ret_raw_b - daily_risk_free_rate
    # just for reference
    annualized_sharpe_ratio_b = daily_ret_b.mean() / daily_ret_b.std() * (T ** 0.5)
    print(f"short QQQ.US annual sharpe: {annualized_sharpe_ratio_b}")

    # combine two RAW rates, then subtract risk-free rate, also normalize it by 2,
    # to calculate rate by capital
    # minus ret_b because we hope it is negative
    daily_ret_ab = (daily_ret_raw_a - daily_ret_raw_b) / 2 - daily_risk_free_rate
    annualized_sharpe_ratio_ab = daily_ret_ab.mean() / daily_ret_ab.std() * (T ** 0.5)
    print(f"long AAPL.US and short QQQ.US annual sharpe: {annualized_sharpe_ratio_ab}")


if __name__ == "__main__":
    sharpe_ratio_example()

