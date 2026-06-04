import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt


# ============================================================
# 0) 一些通用工具函数
# ============================================================
def safe_to_datetime(series: pd.Series) -> pd.Series:
    """把日期列安全转成 datetime（不合法的会变 NaT）"""
    return pd.to_datetime(series, errors="coerce")


def format_percent(value: float, digits: int = 2) -> str:
    """把小数转成百分比字符串"""
    if pd.isna(value):
        return "NaN"
    return f"{value * 100:.{digits}f}%"


def annualized_volatility(daily_return_series: pd.Series, trading_days: int = 252) -> float:
    """用日收益率计算年化波动率（std * sqrt(252)）"""
    if daily_return_series.dropna().empty:
        return np.nan
    return float(daily_return_series.dropna().std(ddof=1) * np.sqrt(trading_days))


def print_divider(title: str):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


# ============================================================
# 1) 美股数据源：用 AkShare 拉美股日线 Close（容错 + 统一格式）
# ============================================================
class USStockData:
    """
    用 akshare 拉美股行情的封装。
    目标：
    - 给 ticker 列表
    - 返回一个对齐后的收盘价矩阵 DataFrame（index=交易日, columns=ticker）
    """

    @staticmethod
    def _download_one_ticker_close(ticker: str) -> pd.Series | None:
        """
        下载单只股票 Close 序列（index=datetime, value=close）。
        AkShare 不同版本接口可能略不同，这里做多种尝试。
        """
        # ✅ 方法 1：stock_us_daily（很多版本都有）
        try:
            df = ak.stock_us_daily(symbol=ticker)
            if df is not None and not df.empty:
                # 兼容字段名：date/日期，close/收盘
                if "date" in df.columns:
                    df["date"] = safe_to_datetime(df["date"])
                    close_col = "close" if "close" in df.columns else None
                    if close_col is None:
                        # 有些版本叫 "收盘"
                        close_col = "收盘" if "收盘" in df.columns else None
                    if close_col is None:
                        return None
                    return df.set_index("date")[close_col].rename(ticker)
        except Exception:
            pass

        # ✅ 方法 2：stock_us_hist（有些版本是这个）
        try:
            df = ak.stock_us_hist(symbol=ticker)
            if df is not None and not df.empty:
                if "日期" in df.columns:
                    df["日期"] = safe_to_datetime(df["日期"])
                    close_col = "收盘" if "收盘" in df.columns else None
                    if close_col is None and "close" in df.columns:
                        close_col = "close"
                    if close_col is None:
                        return None
                    return df.set_index("日期")[close_col].rename(ticker)
        except Exception:
            pass

        return None

    def get_close_price_matrix(
        self,
        ticker_list: list[str],
        start_date: str | None = None,
        end_date: str | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        返回：
        - close_price_df: index=日期, columns=ticker, value=close
        - failed_tickers: 拉取失败的 ticker 列表
        """
        close_series_list: list[pd.Series] = []
        failed_tickers: list[str] = []

        for ticker in ticker_list:
            try:
                series = self._download_one_ticker_close(ticker)
                if series is None or series.dropna().empty:
                    failed_tickers.append(ticker)
                    continue

                # 日期过滤（如果给了 start/end）
                if start_date is not None:
                    start_dt = pd.to_datetime(start_date)
                    series = series[series.index >= start_dt]
                if end_date is not None:
                    end_dt = pd.to_datetime(end_date)
                    series = series[series.index <= end_dt]

                if series.dropna().empty:
                    failed_tickers.append(ticker)
                    continue

                close_series_list.append(series)

            except Exception as e:
                failed_tickers.append(ticker)
                print(f"⚠️ {ticker} 下载失败: {repr(e)}")

        if not close_series_list:
            return pd.DataFrame(), failed_tickers

        close_price_df = pd.concat(close_series_list, axis=1).sort_index()
        close_price_df = close_price_df.dropna(how="all").ffill()

        # 列顺序对齐原列表（只保留成功的）
        successful_tickers = [t for t in ticker_list if t in close_price_df.columns]
        close_price_df = close_price_df.reindex(columns=successful_tickers)

        return close_price_df, failed_tickers


# ============================================================
# 2) 股票池过滤：交易天数 / 价格区间 / 年化波动阈值
# ============================================================
def filter_stock_pool(
    close_price_df: pd.DataFrame,
    minimum_trading_days: int = 504,
    min_price: float = 5.0,
    max_price: float = 150.0,
    max_annual_volatility: float = 0.80
) -> tuple[list[str], pd.DataFrame]:
    """
    根据价格数据过滤股票池：
    - 历史交易日 >= minimum_trading_days
    - 最新价在 [min_price, max_price]
    - 近 252 日年化波动 <= max_annual_volatility
    """
    if close_price_df.empty:
        return [], pd.DataFrame()

    daily_return_df = close_price_df.pct_change(1)

    records = []
    for ticker in close_price_df.columns:
        price_series = close_price_df[ticker].dropna()
        if price_series.empty:
            continue

        trading_days = len(price_series)
        last_price = float(price_series.iloc[-1])

        # 近 252 日年化波动
        one_year_returns = daily_return_df[ticker].dropna().tail(252)
        ann_vol = annualized_volatility(one_year_returns)

        reason_list = []
        if trading_days < minimum_trading_days:
            reason_list.append(f"short_history<{minimum_trading_days}")
        if last_price < min_price:
            reason_list.append(f"low_price<{min_price}")
        if last_price > max_price:
            reason_list.append(f"high_price>{max_price}")
        if pd.notna(ann_vol) and ann_vol > max_annual_volatility:
            reason_list.append(f"high_vol>{max_annual_volatility}")

        records.append({
            "ticker": ticker,
            "trading_days": trading_days,
            "last_price": last_price,
            "ann_vol_12m": ann_vol,
            "reason": ";".join(reason_list) if reason_list else ""
        })

    summary_df = pd.DataFrame(records).sort_values(by=["reason", "ticker"])
    kept_tickers = summary_df[summary_df["reason"] == ""]["ticker"].tolist()

    return kept_tickers, summary_df


# ============================================================
# 3) 日频 -> 月末价格
# ============================================================
def compute_month_end_prices(daily_close_df: pd.DataFrame) -> pd.DataFrame:
    """把日频 Close 压缩成月末 Close（每月最后一个交易日）"""
    if daily_close_df.empty:
        return pd.DataFrame()
    return daily_close_df.resample("ME").last()


# ============================================================
# 4) Residual Volatility 因子（月度）
#    思路：CAPM 回归的残差波动率（idiosyncratic volatility）
#    做法：
#    - 用过去 lookback_days 的日收益率
#    - r_stock = alpha + beta * r_market + residual
#    - residual_vol = std(residual)
# ============================================================
def compute_capm_residual_volatility(
    stock_daily_return: pd.Series,
    market_daily_return: pd.Series
) -> float:
    """
    用简单线性回归（不依赖 statsmodels）估 beta/alpha，然后算残差波动 std。
    beta = cov(r_stock, r_mkt) / var(r_mkt)
    alpha = mean(r_stock) - beta * mean(r_mkt)
    residual = r_stock - (alpha + beta * r_mkt)
    """
    df = pd.concat([stock_daily_return, market_daily_return], axis=1).dropna()
    if df.empty or len(df) < 30:
        return np.nan

    stock = df.iloc[:, 0].values
    market = df.iloc[:, 1].values

    market_var = np.var(market, ddof=1)
    if market_var == 0:
        return np.nan

    beta = np.cov(stock, market, ddof=1)[0, 1] / market_var
    alpha = np.mean(stock) - beta * np.mean(market)

    residual = stock - (alpha + beta * market)
    residual_vol = float(np.std(residual, ddof=1))
    return residual_vol


def build_monthly_residual_vol_factor(
    daily_close_df: pd.DataFrame,
    market_daily_close: pd.Series,
    lookback_days: int = 63
) -> pd.DataFrame:
    """
    输出：factor_monthly_df
    - index = 月末日期
    - columns = ticker
    - value = residual volatility（过去 lookback_days 的日收益率回归残差波动）
    """
    if daily_close_df.empty or market_daily_close.dropna().empty:
        return pd.DataFrame()

    # 日收益率
    stock_daily_return_df = daily_close_df.pct_change(1)
    market_daily_return = market_daily_close.pct_change(1)

    # 月末日期列表
    month_end_dates = compute_month_end_prices(daily_close_df).index.tolist()
    if len(month_end_dates) == 0:
        return pd.DataFrame()

    factor_rows = []

    for month_end_dt in month_end_dates:
        # 截到当月末（避免未来数据）
        stock_return_slice = stock_daily_return_df.loc[:month_end_dt].tail(lookback_days)
        market_return_slice = market_daily_return.loc[:month_end_dt].tail(lookback_days)

        # 市场窗口不够就跳过
        if len(market_return_slice.dropna()) < max(30, lookback_days // 2):
            continue

        factor_values = {}
        for ticker in daily_close_df.columns:
            one_stock_return = stock_return_slice[ticker]
            residual_vol = compute_capm_residual_volatility(one_stock_return, market_return_slice)
            factor_values[ticker] = residual_vol

        factor_rows.append(pd.Series(factor_values, name=month_end_dt))

    if not factor_rows:
        return pd.DataFrame()

    factor_monthly_df = pd.DataFrame(factor_rows).sort_index()
    return factor_monthly_df


# ============================================================
# 5) Rank IC（Spearman）
# ============================================================
def calculate_spearman_rank_ic(
    factor_series: pd.Series,
    future_return_series: pd.Series,
    minimum_stock_count: int = 30
) -> float:
    """
    Rank IC：
    - 在某个时点（比如月末）
    - 看 factor 的横截面排序 与 未来收益的横截面排序相关性
    """
    merged_df = pd.concat([factor_series, future_return_series], axis=1).dropna()
    if len(merged_df) < minimum_stock_count:
        return np.nan

    factor_rank = merged_df.iloc[:, 0].rank()
    return_rank = merged_df.iloc[:, 1].rank()
    return float(factor_rank.corr(return_rank))


def calculate_ic_statistics(ic_series: pd.Series, frequency: int = 12) -> dict:
    """
    常见 IC 指标：
    - IC均值/标准差
    - ICIR（月度） + 年化 ICIR
    - t统计量（均值显著性）
    - 胜率（正IC比例）
    """
    ic_series = ic_series.dropna()
    count = len(ic_series)

    if count == 0:
        return {
            "样本期数(月)": 0,
            "IC均值": np.nan,
            "IC标准差": np.nan,
            "月度ICIR": np.nan,
            "年化ICIR": np.nan,
            "t统计量": np.nan,
            "胜率(正IC比例)": np.nan,
        }

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))

    icir = ic_mean / ic_std if ic_std != 0 else np.nan
    icir_annual = icir * np.sqrt(frequency) if pd.notna(icir) else np.nan

    t_value = ic_mean / (ic_std / np.sqrt(count)) if (ic_std != 0 and count > 1) else np.nan
    win_rate = float((ic_series > 0).mean())

    return {
        "样本期数(月)": count,
        "IC均值": ic_mean,
        "IC标准差": ic_std,
        "月度ICIR": icir,
        "年化ICIR": icir_annual,
        "t统计量": t_value,
        "胜率(正IC比例)": win_rate,
    }


# ============================================================
# 6) 分组：把股票按因子值分 5 组（组内等权）
# ============================================================
def build_group_labels(
    factor_series: pd.Series,
    group_count: int = 5,
    direction: str = "low_is_good"
) -> pd.Series:
    """
    direction:
    - "low_is_good": 因子值越小越好（例如 residual volatility 越低越稳）
      => Group1 = 最低的一组, Group5 = 最高的一组
    - "high_is_good": 因子值越大越好
      => Group1 = 最低, Group5 = 最高（但解释上高更好）
    """
    clean_factor = factor_series.dropna()
    if len(clean_factor) < group_count:
        return pd.Series(dtype=int)

    ranked_values = clean_factor.rank(method="first")

    # qcut：按分位数切组
    group_id = pd.qcut(ranked_values, group_count, labels=False) + 1
    group_id = group_id.astype(int)

    return group_id


def build_equal_weight_group_portfolio(
    factor_series: pd.Series,
    rebalance_date_str: str,
    group_count: int = 5,
    minimum_stock_count: int = 50,
    direction: str = "low_is_good"
) -> pd.DataFrame:
    """
    输出调仓权重表：
    日期、代码、权重、组合名称
    """
    clean_factor = factor_series.dropna()
    if len(clean_factor) < minimum_stock_count:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    group_id_series = build_group_labels(clean_factor, group_count=group_count, direction=direction)
    if group_id_series.empty:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    portfolio_frames = []
    for group_id in range(1, group_count + 1):
        group_tickers = group_id_series.index[group_id_series == group_id].tolist()
        if not group_tickers:
            continue

        equal_weight = 1.0 / len(group_tickers)

        portfolio_frames.append(pd.DataFrame({
            "日期": [rebalance_date_str] * len(group_tickers),
            "代码": group_tickers,
            "权重": [equal_weight] * len(group_tickers),
            "组合名称": [f"Group{group_id}"] * len(group_tickers),
        }))

    if not portfolio_frames:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    return pd.concat(portfolio_frames, ignore_index=True)


# ============================================================
# 7) 回测：月频回测（每月调仓）输出 5 组净值
# ============================================================
def backtest_monthly_groups(
    monthly_close_df: pd.DataFrame,
    monthly_portfolio_df: pd.DataFrame
) -> pd.DataFrame:
    """
    每月调仓回测：
    - monthly_portfolio_df：每月月末的调仓权重（可以每月都有）
    - 输出每组净值（月末序列）
    """
    if monthly_close_df.empty or monthly_portfolio_df.empty:
        return pd.DataFrame()

    portfolio_df = monthly_portfolio_df.copy()
    portfolio_df["rebalance_dt"] = pd.to_datetime(portfolio_df["日期"])

    group_names = sorted(portfolio_df["组合名称"].unique().tolist())
    rebalance_dates = sorted(portfolio_df["rebalance_dt"].unique().tolist())

    # 如果调仓日期少于2个，无法形成收益区间
    if len(rebalance_dates) < 2:
        return pd.DataFrame()

    nav_date_list = [rebalance_dates[0]]
    nav_dict = {name: [1.0] for name in group_names}

    # 月度收益：从上月末到本月末
    monthly_return_df = monthly_close_df.pct_change(1)

    for i in range(len(rebalance_dates) - 1):
        current_dt = rebalance_dates[i]
        next_dt = rebalance_dates[i + 1]

        if current_dt not in monthly_close_df.index or next_dt not in monthly_close_df.index:
            continue

        holdings_df = portfolio_df[portfolio_df["rebalance_dt"] == current_dt]
        one_month_ret = monthly_return_df.loc[next_dt]  # 这一期的收益（从 current_dt -> next_dt）

        for group_name in group_names:
            group_holdings = holdings_df[holdings_df["组合名称"] == group_name]
            tickers = group_holdings["代码"].tolist()
            weights = group_holdings.set_index("代码")["权重"]

            aligned_ret = one_month_ret.reindex(tickers).fillna(0.0)
            aligned_w = weights.reindex(tickers).fillna(0.0)

            group_return = float((aligned_ret * aligned_w).sum())
            latest_nav = nav_dict[group_name][-1]
            nav_dict[group_name].append(latest_nav * (1.0 + group_return))

        nav_date_list.append(next_dt)

    nav_df = pd.DataFrame(nav_dict, index=nav_date_list)
    nav_df.index.name = "date"
    return nav_df


# ============================================================
# 8) 赢家组诊断 + 推荐表
# ============================================================
def diagnose_recent_winner_groups(nav_df: pd.DataFrame, recent_months: int = 12) -> pd.Series:
    """
    统计最近 N 个月里，每个月净值最高的组是谁
    """
    if nav_df.empty or len(nav_df) < 2:
        return pd.Series(dtype=int)

    recent_nav = nav_df.tail(recent_months)
    monthly_winner_group = recent_nav.idxmax(axis=1)
    return monthly_winner_group.value_counts()


def build_recommendation_table(
    as_of_month_end: pd.Timestamp,
    winner_group: str,
    factor_monthly_df: pd.DataFrame,
    monthly_close_df: pd.DataFrame,
    daily_close_df: pd.DataFrame,
    direction: str = "low_is_good",
    top_n: int = 10
) -> pd.DataFrame:
    """
    推荐表：
    - 最新一期（月末）因子 -> 分组 -> 取赢家组
    - 在赢家组内部按因子排序取 top_n（低更好就取最小）
    - 输出：ticker, residual_vol, ret_3m, ret_12m, ann_vol_12m, weight_eq
    """
    if factor_monthly_df.empty or as_of_month_end not in factor_monthly_df.index:
        return pd.DataFrame()

    factor_today = factor_monthly_df.loc[as_of_month_end].dropna()
    if factor_today.empty:
        return pd.DataFrame()

    # 分组标签
    group_id_series = build_group_labels(factor_today, group_count=5, direction=direction)
    if group_id_series.empty:
        return pd.DataFrame()

    target_group_id = int(winner_group.replace("Group", ""))
    winner_group_tickers = group_id_series.index[group_id_series == target_group_id].tolist()
    if not winner_group_tickers:
        return pd.DataFrame()

    # 因子方向：低好 -> 升序取最小；高好 -> 降序取最大
    sort_ascending = True if direction == "low_is_good" else False
    candidate_factor = factor_today.reindex(winner_group_tickers).dropna()
    candidate_factor = candidate_factor.sort_values(ascending=sort_ascending).head(top_n)

    selected_tickers = candidate_factor.index.tolist()
    if len(selected_tickers) == 0:
        return pd.DataFrame()

    # 3m / 12m 收益（用月末价做）
    ret_3m_series = (monthly_close_df / monthly_close_df.shift(3) - 1.0).loc[as_of_month_end]
    ret_12m_series = (monthly_close_df / monthly_close_df.shift(12) - 1.0).loc[as_of_month_end]

    # 12m 年化波动（用最近 252 日日收益率）
    daily_ret_df = daily_close_df.pct_change(1)
    ann_vol_series = daily_ret_df.tail(252).std(ddof=1) * np.sqrt(252)

    equal_weight = 1.0 / len(selected_tickers)

    report_df = pd.DataFrame({
        "as_of_month_end": [as_of_month_end] * len(selected_tickers),
        "winner_group_from_backtest": [winner_group] * len(selected_tickers),
        "ticker": selected_tickers,
        "group": [winner_group] * len(selected_tickers),
        "weight_eq": [equal_weight] * len(selected_tickers),
        "residual_vol": candidate_factor.values,
        "ret_3m": ret_3m_series.reindex(selected_tickers).values,
        "ret_12m": ret_12m_series.reindex(selected_tickers).values,
        "ann_vol_12m": ann_vol_series.reindex(selected_tickers).values,
        "rule": [f"Winner group = {winner_group} -> pick top_n within group"] * len(selected_tickers)
    })

    return report_df


# ============================================================
# 9) 画图模块：IC + 组净值 + 多空净值
# ============================================================
def plot_monthly_ic_report(ic_series: pd.Series, title_prefix: str = "Monthly Rank IC"):
    """3张图：IC柱状 + IC折线 + 累计IC"""
    ic_series = ic_series.dropna().sort_index()
    if ic_series.empty:
        return

    cumulative_ic = ic_series.cumsum()

    # 1) Bar
    plt.figure()
    plt.bar(ic_series.index, ic_series.values, width=20)
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Bar)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 2) Line
    plt.figure()
    plt.plot(ic_series.index, ic_series.values, marker="o")
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Line)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 3) Cumulative
    plt.figure()
    plt.plot(cumulative_ic.index, cumulative_ic.values, marker="o")
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Cumulative)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_group_nav(nav_df: pd.DataFrame, title: str = "Group NAV (Monthly Rebalance)"):
    """5组净值曲线"""
    if nav_df.empty:
        return

    plt.figure()
    for group_name in nav_df.columns:
        plt.plot(nav_df.index, nav_df[group_name], label=group_name)

    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_long_short_nav(nav_df: pd.DataFrame):
    """多空净值：Group5 / Group1（是否有明显分层一眼能看）"""
    if nav_df.empty:
        return
    if "Group5" not in nav_df.columns or "Group1" not in nav_df.columns:
        return

    long_short_nav = nav_df["Group5"] / nav_df["Group1"]

    plt.figure()
    plt.plot(long_short_nav.index, long_short_nav.values)
    plt.title("Long-Short NAV (Group5 / Group1)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================
# 10) main：全流程串联
# ============================================================
def main():
    # --------------------------------------------------------
    # A) 股票池
    # --------------------------------------------------------
    base_ticker_list = [
        "GE", "GEV", "ROBT", "AIQ", "CGNX", "TER", "MU", "TSM", "AMD", "AAPL", "INTC",
        "BMNR", "ISRG", "IAU", "NVDA", "TSLA", "NTD0Y", "PLTR", "SONY", "PFE", "DJT",
        "AMZN", "MSTR", "COIN", "BURBY", "JANX", "ASST", "AMZE", "IBRX", "SLGB", "GITS",
        "DVLT", "NU", "GORO","GLD",

        # 扩充池
        "SPLG", "QCOM", "TXN", "AMAT", "ADI", "CSCO", "DELL", "HPQ", "UBER",
        "BAC", "WFC", "C", "SCHW", "COF", "USB",
        "MRK", "BMY", "GILD", "CVS",
        "KO", "TGT", "NKE", "SBUX", "DIS", "EBAY",
        "OXY", "SLB", "COP", "VLO",
        "DUK", "SO", "EXC","STX",
        "O", "VICI", "WDC", "NTAP"
    ]

    # --------------------------------------------------------
    # B) 下载价格数据
    # --------------------------------------------------------
    print_divider("用 Akshare 下载美股日线（Close）")
    print("Ticker List:", base_ticker_list)

    data_source = USStockData()
    close_price_df, failed_tickers = data_source.get_close_price_matrix(base_ticker_list)

    if failed_tickers:
        print("\n⚠️ 无数据/失败 ticker：", failed_tickers)

    if close_price_df.empty:
        print("\n❌ 没拉到任何价格数据，结束。")
        return

    # --------------------------------------------------------
    # C) 下载市场基准（SPY）
    # --------------------------------------------------------
    print_divider("用 Akshare 下载美股日线（Close）")
    print("Ticker List:", ["SPY"])
    market_close_df, failed_market = data_source.get_close_price_matrix(["SPY"])
    if market_close_df.empty:
        print("\n❌ SPY 数据下载失败，无法计算 Residual Volatility。")
        return

    market_close_series = market_close_df["SPY"].dropna()

    # --------------------------------------------------------
    # D) 股票池过滤（价格版）
    # --------------------------------------------------------
    print_divider("股票池过滤（价格数据版）")
    filter_rule_text = "历史≥504天；年化波动≤0.80；5.0≤最新价≤150.0"
    print("规则：", filter_rule_text)

    kept_tickers, filter_summary_df = filter_stock_pool(
        close_price_df,
        minimum_trading_days=504,
        min_price=5.0,
        max_price=150.0,
        max_annual_volatility=0.80
    )

    print(f"原始ticker数: {len(close_price_df.columns)}")
    print(f"保留ticker数: {len(kept_tickers)}")
    print(f"剔除ticker数: {len(close_price_df.columns) - len(kept_tickers)}")

    removed_df = filter_summary_df[filter_summary_df["reason"] != ""]
    if not removed_df.empty:
        print("\n--- 剔除清单（前25条）---")
        print(removed_df.head(25)[["ticker", "reason", "trading_days", "last_price", "ann_vol_12m"]].to_string(index=False))

    if len(kept_tickers) < 10:
        print("\n❌ 过滤后可用ticker太少（<10），建议：放宽阈值或扩大股票池。")
        return

    filtered_close_df = close_price_df[kept_tickers].copy()

    # --------------------------------------------------------
    # E) 计算 Residual Volatility 月度因子
    # --------------------------------------------------------
    print_divider("开始 Residual Volatility 回测（Monthly + 最终整合版）")

    # 你可以改 lookback_days：
    # - 21：1个月
    # - 63：3个月（比较稳）
    # - 126：半年
    lookback_days = 63

    factor_monthly_df = build_monthly_residual_vol_factor(
        daily_close_df=filtered_close_df,
        market_daily_close=market_close_series,
        lookback_days=lookback_days
    )

    if factor_monthly_df.empty:
        print("\n❌ Residual Volatility 因子为空（可能 market 对齐失败/历史太短），结束。")
        return

    # --------------------------------------------------------
    # F) 月末价格 -> 月收益（用于未来收益）
    # --------------------------------------------------------
    monthly_close_df = compute_month_end_prices(filtered_close_df)

    # 因子月末日期，要跟 monthly_close 对齐
    common_month_end_dates = factor_monthly_df.index.intersection(monthly_close_df.index)
    factor_monthly_df = factor_monthly_df.loc[common_month_end_dates]
    monthly_close_df = monthly_close_df.loc[common_month_end_dates]

    # 未来1个月收益（标准月度因子检验方式）
    future_1m_return_df = monthly_close_df.pct_change(1).shift(-1)

    # --------------------------------------------------------
    # G) Rank IC（每个月算一次）
    # --------------------------------------------------------
    minimum_ic_stock_count = 20  # 真实建议更高，池子小就调低
    ic_records = []

    for month_end_dt in factor_monthly_df.index:
        factor_series = factor_monthly_df.loc[month_end_dt]
        future_ret_series = future_1m_return_df.loc[month_end_dt]

        rank_ic = calculate_spearman_rank_ic(
            factor_series=factor_series,
            future_return_series=future_ret_series,
            minimum_stock_count=minimum_ic_stock_count
        )
        ic_records.append((month_end_dt, rank_ic))

    ic_series = pd.Series(
        [ic for _, ic in ic_records],
        index=[dt for dt, _ in ic_records],
        name="RankIC_ResidualVol_Monthly"
    ).dropna()

    print_divider("IC 统计")
    stats = calculate_ic_statistics(ic_series, frequency=12)
    for key, value in stats.items():
        if isinstance(value, float):
            if "胜率" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # IC 图表（3张）
    plot_monthly_ic_report(ic_series, title_prefix="Monthly Rank IC (Residual Volatility Factor)")

    # --------------------------------------------------------
    # H) 分组建仓（每月调仓） + 5组净值回测
    # --------------------------------------------------------
    group_count = 5
    direction = "low_is_good"  # residual vol 越低越好（一般逻辑）
    minimum_group_stock_count = 20

    portfolio_list = []
    for month_end_dt in factor_monthly_df.index:
        factor_this_month = factor_monthly_df.loc[month_end_dt]

        portfolio_df = build_equal_weight_group_portfolio(
            factor_series=factor_this_month,
            rebalance_date_str=month_end_dt.strftime("%Y%m%d"),
            group_count=group_count,
            minimum_stock_count=minimum_group_stock_count,
            direction=direction
        )

        if not portfolio_df.empty:
            portfolio_list.append(portfolio_df)

    if not portfolio_list:
        print("\n❌ portfolio 为空：说明可用股票太少或因子缺失太多。")
        return

    monthly_portfolio_df = pd.concat(portfolio_list, ignore_index=True)

    # 调仓日必须在月末价格里
    monthly_portfolio_df["rebalance_dt"] = pd.to_datetime(monthly_portfolio_df["日期"])
    monthly_portfolio_df = monthly_portfolio_df[monthly_portfolio_df["rebalance_dt"].isin(monthly_close_df.index)]

    nav_df = backtest_monthly_groups(
        monthly_close_df=monthly_close_df,
        monthly_portfolio_df=monthly_portfolio_df[["日期", "代码", "权重", "组合名称"]]
    )

    print_divider("净值（最后5行）")
    if nav_df.empty:
        print("Empty NAV dataframe.")
        return
    print(nav_df.tail())

    # 组净值图 + 多空图（2张）
    plot_group_nav(nav_df, title="Group NAV (Residual Volatility Factor, Monthly Rebalance)")
    plot_long_short_nav(nav_df)

    # --------------------------------------------------------
    # I) 诊断：最近12个月赢家组频次
    # --------------------------------------------------------
    winner_count = diagnose_recent_winner_groups(nav_df, recent_months=12)
    print_divider("诊断：最近赢家组频次（近12个月）")
    if winner_count.empty:
        print("⚠️ 无法统计赢家组（nav_df 为空或月份不足）")
        return
    print(winner_count.to_string())

    winner_group = str(winner_count.index[0])  # 出现最多次的赢家组

    # --------------------------------------------------------
    # J) 最新一期推荐（赢家组）
    # --------------------------------------------------------
    latest_month_end = factor_monthly_df.index.max()

    recommend_df = build_recommendation_table(
        as_of_month_end=latest_month_end,
        winner_group=winner_group,
        factor_monthly_df=factor_monthly_df,
        monthly_close_df=monthly_close_df,
        daily_close_df=filtered_close_df,
        direction=direction,
        top_n=10
    )

    print_divider("最新一期推荐（赢家组）")
    if recommend_df.empty:
        print("⚠️ 推荐表为空：可能赢家组当期没有足够股票 / 因子缺失。")
        return

    print(f"调仓月末: {latest_month_end.date()}  |  赢家组: {winner_group}  |  等权持仓数: {len(recommend_df)}")
    print("Tickers:", recommend_df["ticker"].tolist())

    print_divider("推荐表（基于：回测赢家组 + 最新一期因子分组）")
    # 把收益字段显示得更友好
    display_df = recommend_df.copy()
    display_df["weight_eq"] = display_df["weight_eq"].apply(lambda x: format_percent(x, 2))
    display_df["ret_3m"] = display_df["ret_3m"].apply(lambda x: format_percent(x, 2))
    display_df["ret_12m"] = display_df["ret_12m"].apply(lambda x: format_percent(x, 2))
    display_df["ann_vol_12m"] = display_df["ann_vol_12m"].apply(lambda x: format_percent(x, 2))
    print(display_df.to_string(index=False))

    # --------------------------------------------------------
    # K) 可选：保存结果到 CSV
    # --------------------------------------------------------
    # ic_series.to_csv("ic_series.csv", encoding="utf-8-sig")
    # nav_df.to_csv("group_nav.csv", encoding="utf-8-sig")
    # recommend_df.to_csv("recommendation_table.csv", index=False, encoding="utf-8-sig")
    # print("\n✅ 已保存：ic_series.csv / group_nav.csv / recommendation_table.csv")


if __name__ == "__main__":
    main()
