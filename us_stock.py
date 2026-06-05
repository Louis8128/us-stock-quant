from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import akshare as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 0) 通用工具
# ============================================================
def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a date-like series to pandas datetime and coerce invalid values to NaT."""
    return pd.to_datetime(series, errors="coerce")


def format_percent(value: float, digits: int = 2) -> str:
    """Format a decimal return/weight as percentage text."""
    if pd.isna(value):
        return "NaN"
    return f"{value * 100:.{digits}f}%"


def annualized_volatility(daily_return_series: pd.Series, trading_days: int = 252) -> float:
    """Calculate annualized volatility from daily returns."""
    clean_return = daily_return_series.dropna()
    if clean_return.empty:
        return np.nan
    return float(clean_return.std(ddof=1) * np.sqrt(trading_days))


def print_divider(title: str) -> None:
    """Print a readable console section divider."""
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def memory_usage_mb(frame: pd.DataFrame) -> float:
    """Return deep memory usage of a DataFrame in MB."""
    if frame.empty:
        return 0.0
    return float(frame.memory_usage(deep=True).sum() / 1024**2)


def downcast_float_frame(frame: pd.DataFrame, dtype: str = "float32") -> pd.DataFrame:
    """
    Downcast numeric DataFrame values to reduce memory usage.

    float32 is usually enough for daily prices, returns and factor values in this
    script. If you need very high precision accounting numbers, pass dtype="float64".
    """
    if frame.empty:
        return frame
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    return numeric_frame.astype(dtype, copy=False)


def optimize_portfolio_memory(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Use compact dtypes for the long-format portfolio table."""
    if portfolio_df.empty:
        return portfolio_df

    optimized_df = portfolio_df.copy()
    for col in ["代码", "组合名称"]:
        if col in optimized_df.columns:
            optimized_df[col] = optimized_df[col].astype("category")
    if "权重" in optimized_df.columns:
        optimized_df["权重"] = pd.to_numeric(optimized_df["权重"], errors="coerce").astype("float32")
    return optimized_df


def resample_month_end_last(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Resample to month-end using the last available observation.

    Pandas 2.2 prefers "ME"; older versions may only support "M", so this keeps
    the script compatible across common research environments.
    """
    try:
        return data.resample("ME").last()
    except ValueError:
        return data.resample("M").last()


# ============================================================
# 1) 参数配置
# ============================================================
@dataclass(frozen=True)
class DataDownloadConfig:
    """Configuration for AkShare downloads and local cache behavior."""

    cache_dir: str | Path = field(default_factory=lambda: Path("data_cache") / "us_stock")
    use_cache: bool = True
    force_refresh: bool = False
    cache_stale_days: int | None = 1
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5
    max_workers: int = 4
    verbose: bool = True


@dataclass(frozen=True)
class ResidualVolatilityBacktestConfig:
    """Configuration for filtering, factor calculation, IC test and group backtest."""

    start_date: str | None = None
    end_date: str | None = None
    benchmark_ticker: str = "SPY"
    minimum_trading_days: int = 504
    min_price: float = 5.0
    max_price: float = 150.0
    max_annual_volatility: float = 0.80
    lookback_days: int = 63
    minimum_regression_observations: int = 30
    group_count: int = 5
    direction: str = "low_is_good"
    minimum_ic_stock_count: int = 20
    minimum_group_stock_count: int = 20
    recommendation_top_n: int = 10
    recent_winner_months: int = 12


@dataclass
class BacktestResult:
    """Container for major intermediate and final outputs of the workflow."""

    close_price_df: pd.DataFrame
    filtered_close_df: pd.DataFrame
    market_close_series: pd.Series
    filter_summary_df: pd.DataFrame
    factor_monthly_df: pd.DataFrame
    ic_series: pd.Series
    ic_statistics: dict[str, float | int]
    monthly_portfolio_df: pd.DataFrame
    nav_df: pd.DataFrame
    recommendation_df: pd.DataFrame


# ============================================================
# 2) 美股数据源：AkShare + Retry + 本地缓存
# ============================================================
class USStockData:
    """
    Download US stock close prices from AkShare with retry and local cache.

    Returned close price matrix:
    - index: trading date
    - columns: ticker
    - values: close price
    """

    def __init__(self, config: DataDownloadConfig | None = None) -> None:
        self.config = config or DataDownloadConfig()
        self.cache_dir = Path(self.config.cache_dir)

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:
        """Normalize ticker text for AkShare calls and cache file names."""
        return str(ticker).strip().upper()

    def _cache_path(self, ticker: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._sanitize_ticker(ticker))
        return self.cache_dir / f"{safe_name}.pkl"

    def _cache_is_fresh(self, ticker: str) -> bool:
        """Return True when a cache file exists and is not stale."""
        path = self._cache_path(ticker)
        if not path.exists():
            return False
        if self.config.cache_stale_days is None:
            return True
        max_age_seconds = self.config.cache_stale_days * 24 * 60 * 60
        return (time.time() - path.stat().st_mtime) <= max_age_seconds

    def _read_cached_ticker_close(self, ticker: str) -> pd.Series | None:
        """Read a cached ticker close series if available and valid."""
        path = self._cache_path(ticker)
        if not path.exists():
            return None

        try:
            cached_obj = pd.read_pickle(path)
            if isinstance(cached_obj, pd.Series):
                series = cached_obj.copy()
            elif isinstance(cached_obj, pd.DataFrame):
                close_col = "close" if "close" in cached_obj.columns else cached_obj.columns[0]
                series = cached_obj[close_col].copy()
            else:
                return None

            series.index = pd.to_datetime(series.index, errors="coerce")
            series = pd.to_numeric(series, errors="coerce")
            series = series[series.index.notna()].dropna()
            if series.empty:
                return None

            series = series[~series.index.duplicated(keep="last")].sort_index()
            return series.astype("float32").rename(self._sanitize_ticker(ticker))
        except Exception as exc:
            if self.config.verbose:
                print(f"{ticker} cache read failed: {repr(exc)}")
            return None

    def _write_cached_ticker_close(self, ticker: str, series: pd.Series) -> None:
        """Persist a ticker close series to local pickle cache."""
        if not self.config.use_cache or series.empty:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            clean_series = pd.to_numeric(series, errors="coerce").dropna().astype("float32")
            clean_series.to_pickle(self._cache_path(ticker))
        except Exception as exc:
            if self.config.verbose:
                print(f"{ticker} cache write failed: {repr(exc)}")

    @staticmethod
    def _normalize_akshare_close_frame(df: pd.DataFrame, ticker: str) -> pd.Series | None:
        """Normalize different AkShare schemas into one close-price Series."""
        if df is None or df.empty:
            return None

        lower_to_original = {str(col).lower(): col for col in df.columns}
        date_candidates = ["date", "日期", "Date", "时间", "datetime"]
        close_candidates = ["close", "收盘", "Close", "adj_close", "adjusted_close"]

        date_col = next((col for col in date_candidates if col in df.columns), None)
        if date_col is None:
            date_col = next((lower_to_original[col.lower()] for col in date_candidates if col.lower() in lower_to_original), None)

        close_col = next((col for col in close_candidates if col in df.columns), None)
        if close_col is None:
            close_col = next((lower_to_original[col.lower()] for col in close_candidates if col.lower() in lower_to_original), None)

        if date_col is None or close_col is None:
            return None

        normalized_df = df[[date_col, close_col]].copy()
        normalized_df[date_col] = safe_to_datetime(normalized_df[date_col])
        normalized_df[close_col] = pd.to_numeric(normalized_df[close_col], errors="coerce")
        normalized_df = normalized_df.dropna(subset=[date_col, close_col])
        if normalized_df.empty:
            return None

        series = normalized_df.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)[close_col]
        series = series.sort_index()
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_convert(None)

        return series.astype("float32").rename(ticker)

    def _fetch_from_akshare(self, ticker: str) -> pd.Series | None:
        """
        Fetch one ticker from AkShare.

        AkShare has used different functions/schemas across versions, so this
        method tries the two common US equity endpoints and normalizes the result.
        """
        fetchers = [
            ("stock_us_daily", lambda: ak.stock_us_daily(symbol=ticker)),
            ("stock_us_hist", lambda: ak.stock_us_hist(symbol=ticker)),
        ]
        errors: list[str] = []

        for name, fetcher in fetchers:
            try:
                raw_df = fetcher()
                series = self._normalize_akshare_close_frame(raw_df, ticker)
                if series is not None and not series.empty:
                    return series
                errors.append(f"{name}: empty or unsupported schema")
            except Exception as exc:
                errors.append(f"{name}: {repr(exc)}")

        if errors:
            raise RuntimeError("; ".join(errors))
        return None

    def download_ticker_close(self, ticker: str) -> pd.Series | None:
        """
        Download one ticker close series with cache-first, retry and stale-cache fallback.

        Cache rules:
        - fresh cache and force_refresh=False: return cache directly
        - stale/missing cache: try AkShare with retry
        - AkShare still fails: return stale cache if available
        """
        ticker = self._sanitize_ticker(ticker)
        cached_series = self._read_cached_ticker_close(ticker) if self.config.use_cache else None

        if (
            self.config.use_cache
            and not self.config.force_refresh
            and cached_series is not None
            and self._cache_is_fresh(ticker)
        ):
            return cached_series

        last_error: Exception | None = None
        max_retries = max(1, self.config.max_retries)

        for attempt in range(1, max_retries + 1):
            try:
                series = self._fetch_from_akshare(ticker)
                if series is not None and not series.dropna().empty:
                    self._write_cached_ticker_close(ticker, series)
                    return series
                last_error = RuntimeError("empty close series")
            except Exception as exc:
                last_error = exc

            if attempt < max_retries:
                sleep_seconds = self.config.retry_sleep_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

        if cached_series is not None:
            if self.config.verbose:
                print(f"{ticker} download failed, using stale cache. Last error: {repr(last_error)}")
            return cached_series

        if self.config.verbose:
            print(f"{ticker} download failed. Last error: {repr(last_error)}")
        return None

    @staticmethod
    def _clip_by_date(
        series: pd.Series,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Filter a price series by optional inclusive start/end dates."""
        clipped = series
        if start_date is not None:
            clipped = clipped[clipped.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            clipped = clipped[clipped.index <= pd.to_datetime(end_date)]
        return clipped

    def get_close_price_matrix(
        self,
        ticker_list: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Download and align ticker close prices.

        Returns
        -------
        close_price_df:
            DataFrame with index=date, columns=ticker and value=close.
        failed_tickers:
            Tickers with no usable data after download/cache/date filtering.
        """
        normalized_tickers = []
        seen_tickers: set[str] = set()
        for ticker in ticker_list:
            normalized = self._sanitize_ticker(ticker)
            if normalized and normalized not in seen_tickers:
                normalized_tickers.append(normalized)
                seen_tickers.add(normalized)

        if not normalized_tickers:
            return pd.DataFrame(), []

        close_by_ticker: dict[str, pd.Series] = {}
        failed_tickers: list[str] = []

        def load_one(ticker: str) -> tuple[str, pd.Series | None]:
            series = self.download_ticker_close(ticker)
            if series is not None:
                series = self._clip_by_date(series, start_date=start_date, end_date=end_date)
            return ticker, series

        max_workers = min(max(1, self.config.max_workers), len(normalized_tickers))
        if max_workers == 1:
            for ticker in normalized_tickers:
                ticker, series = load_one(ticker)
                if series is None or series.dropna().empty:
                    failed_tickers.append(ticker)
                else:
                    close_by_ticker[ticker] = series.rename(ticker)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(load_one, ticker): ticker for ticker in normalized_tickers}
                for future in as_completed(future_map):
                    ticker = future_map[future]
                    try:
                        _, series = future.result()
                    except Exception as exc:
                        if self.config.verbose:
                            print(f"{ticker} download task failed: {repr(exc)}")
                        failed_tickers.append(ticker)
                        continue

                    if series is None or series.dropna().empty:
                        failed_tickers.append(ticker)
                    else:
                        close_by_ticker[ticker] = series.rename(ticker)

        if not close_by_ticker:
            return pd.DataFrame(), failed_tickers

        close_price_df = pd.concat(
            [close_by_ticker[ticker] for ticker in normalized_tickers if ticker in close_by_ticker],
            axis=1,
        ).sort_index()
        close_price_df = close_price_df[~close_price_df.index.duplicated(keep="last")]
        close_price_df = close_price_df.dropna(how="all").ffill()
        close_price_df = close_price_df.reindex(columns=[ticker for ticker in normalized_tickers if ticker in close_price_df.columns])

        return downcast_float_frame(close_price_df), failed_tickers


# ============================================================
# 3) 股票池过滤
# ============================================================
def filter_stock_pool(
    close_price_df: pd.DataFrame,
    minimum_trading_days: int = 504,
    min_price: float = 5.0,
    max_price: float = 150.0,
    max_annual_volatility: float = 0.80,
) -> tuple[list[str], pd.DataFrame]:
    """
    Filter the stock pool by history length, latest price and recent volatility.

    The numeric statistics are vectorized. Only the human-readable reason string
    uses a light ticker loop, which is not performance critical.
    """
    if close_price_df.empty:
        return [], pd.DataFrame()

    clean_close_df = downcast_float_frame(close_price_df).sort_index()
    daily_return_df = clean_close_df.pct_change(fill_method=None)

    trading_days = clean_close_df.notna().sum(axis=0)
    last_price = clean_close_df.ffill().iloc[-1]
    ann_vol_12m = daily_return_df.tail(252).std(ddof=1) * np.sqrt(252)

    summary_df = pd.DataFrame(
        {
            "ticker": clean_close_df.columns,
            "trading_days": trading_days.reindex(clean_close_df.columns).astype("int32").values,
            "last_price": last_price.reindex(clean_close_df.columns).astype("float32").values,
            "ann_vol_12m": ann_vol_12m.reindex(clean_close_df.columns).astype("float32").values,
        }
    )

    reasons: list[str] = []
    for row in summary_df.itertuples(index=False):
        reason_list: list[str] = []
        if row.trading_days < minimum_trading_days:
            reason_list.append(f"short_history<{minimum_trading_days}")
        if row.last_price < min_price:
            reason_list.append(f"low_price<{min_price}")
        if row.last_price > max_price:
            reason_list.append(f"high_price>{max_price}")
        if pd.notna(row.ann_vol_12m) and row.ann_vol_12m > max_annual_volatility:
            reason_list.append(f"high_vol>{max_annual_volatility}")
        reasons.append(";".join(reason_list))

    summary_df["reason"] = reasons
    summary_df = summary_df.sort_values(by=["reason", "ticker"]).reset_index(drop=True)
    kept_tickers = summary_df.loc[summary_df["reason"] == "", "ticker"].tolist()

    return kept_tickers, summary_df


# ============================================================
# 4) 月末价格
# ============================================================
def compute_month_end_prices(daily_close_df: pd.DataFrame) -> pd.DataFrame:
    """Compress daily close prices to month-end close prices."""
    if daily_close_df.empty:
        return pd.DataFrame()
    monthly_close_df = resample_month_end_last(daily_close_df).dropna(how="all")
    return downcast_float_frame(monthly_close_df)


# ============================================================
# 5) Residual Volatility 因子
# ============================================================
def compute_capm_residual_volatility(
    stock_daily_return: pd.Series,
    market_daily_return: pd.Series,
) -> float:
    """
    Compute single-stock CAPM residual volatility.

    This function is kept for diagnostics and unit checks. The production factor
    path uses compute_rolling_capm_residual_volatility_matrix for speed.
    """
    df = pd.concat([stock_daily_return, market_daily_return], axis=1).dropna()
    if df.empty or len(df) < 30:
        return np.nan

    stock = df.iloc[:, 0].to_numpy(dtype=np.float64)
    market = df.iloc[:, 1].to_numpy(dtype=np.float64)

    market_var = np.var(market, ddof=1)
    if market_var == 0:
        return np.nan

    beta = np.cov(stock, market, ddof=1)[0, 1] / market_var
    alpha = np.mean(stock) - beta * np.mean(market)
    residual = stock - (alpha + beta * market)
    return float(np.std(residual, ddof=1))


def compute_rolling_capm_residual_volatility_matrix(
    stock_daily_return_df: pd.DataFrame,
    market_daily_return: pd.Series,
    lookback_days: int = 63,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """
    Vectorized rolling CAPM residual volatility for all tickers.

    For each rolling window and ticker, the function estimates:
        r_stock = alpha + beta * r_market + residual

    It avoids the slow month-end x ticker regression loop by using rolling sums:
        Sxx, Syy, Sxy -> beta -> SSE -> residual std

    The output matches the previous convention of std(residual, ddof=1), not the
    classical regression sigma with n-2 degrees of freedom.
    """
    if stock_daily_return_df.empty or market_daily_return.dropna().empty:
        return pd.DataFrame()
    if lookback_days < 3:
        raise ValueError("lookback_days must be >= 3")

    min_obs = max(3, minimum_observations)
    y = downcast_float_frame(stock_daily_return_df).sort_index()
    x = pd.to_numeric(market_daily_return.reindex(y.index), errors="coerce").astype("float32")

    valid = y.notna().mul(x.notna(), axis=0).astype(bool)
    valid_float = valid.astype("float32")
    x0 = x.fillna(0.0).astype("float32")
    y0 = y.where(valid, 0.0).astype("float32")

    rolling_window = {"window": lookback_days, "min_periods": 1}
    n = valid_float.rolling(**rolling_window).sum()
    sum_x = valid_float.mul(x0, axis=0).rolling(**rolling_window).sum()
    sum_x2 = valid_float.mul(x0 * x0, axis=0).rolling(**rolling_window).sum()
    sum_y = y0.rolling(**rolling_window).sum()
    sum_y2 = (y0 * y0).rolling(**rolling_window).sum()
    sum_xy = y0.mul(x0, axis=0).rolling(**rolling_window).sum()

    n_safe = n.where(n > 0)
    sxx = sum_x2 - (sum_x * sum_x) / n_safe
    syy = sum_y2 - (sum_y * sum_y) / n_safe
    sxy = sum_xy - (sum_x * sum_y) / n_safe

    beta = sxy / sxx.replace(0.0, np.nan)
    sse = syy - beta * sxy
    residual_var = sse / (n_safe - 1.0)

    valid_result = (n >= min_obs) & (sxx > 1e-12)
    residual_var = residual_var.where(valid_result).clip(lower=0.0)
    residual_vol_df = np.sqrt(residual_var)

    return downcast_float_frame(residual_vol_df)


def build_monthly_residual_vol_factor(
    daily_close_df: pd.DataFrame,
    market_daily_close: pd.Series,
    lookback_days: int = 63,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """
    Build monthly residual volatility factor from daily stock and market closes.

    Returns a DataFrame with:
    - index: month-end trading date
    - columns: ticker
    - values: trailing CAPM residual volatility
    """
    if daily_close_df.empty or market_daily_close.dropna().empty:
        return pd.DataFrame()

    daily_close_df = downcast_float_frame(daily_close_df).sort_index()
    market_daily_close = pd.to_numeric(market_daily_close.sort_index(), errors="coerce")

    stock_daily_return_df = daily_close_df.pct_change(fill_method=None)
    market_daily_return = market_daily_close.pct_change(fill_method=None).reindex(stock_daily_return_df.index)

    residual_vol_daily_df = compute_rolling_capm_residual_volatility_matrix(
        stock_daily_return_df=stock_daily_return_df,
        market_daily_return=market_daily_return,
        lookback_days=lookback_days,
        minimum_observations=minimum_observations,
    )
    if residual_vol_daily_df.empty:
        return pd.DataFrame()

    month_end_dates = compute_month_end_prices(daily_close_df).index
    factor_monthly_df = residual_vol_daily_df.reindex(month_end_dates).dropna(how="all")
    return downcast_float_frame(factor_monthly_df)


# ============================================================
# 6) Rank IC
# ============================================================
def calculate_spearman_rank_ic(
    factor_series: pd.Series,
    future_return_series: pd.Series,
    minimum_stock_count: int = 30,
) -> float:
    """
    Calculate one cross-sectional Spearman Rank IC value.

    Rank IC is the correlation between cross-sectional factor ranks and future
    return ranks at a single rebalance date.
    """
    merged_df = pd.concat([factor_series, future_return_series], axis=1).dropna()
    if len(merged_df) < minimum_stock_count:
        return np.nan

    factor_rank = merged_df.iloc[:, 0].rank()
    return_rank = merged_df.iloc[:, 1].rank()
    return float(factor_rank.corr(return_rank))


def calculate_monthly_rank_ic(
    factor_monthly_df: pd.DataFrame,
    future_return_df: pd.DataFrame,
    minimum_stock_count: int = 30,
) -> pd.Series:
    """
    Vectorized monthly Rank IC calculation for all rebalance dates.

    This replaces the explicit month loop with row-wise DataFrame ranking and
    corrwith(axis=1).
    """
    if factor_monthly_df.empty or future_return_df.empty:
        return pd.Series(dtype="float64", name="RankIC_ResidualVol_Monthly")

    factor_df, future_df = factor_monthly_df.align(future_return_df, join="inner", axis=0)
    factor_df, future_df = factor_df.align(future_df, join="inner", axis=1)
    if factor_df.empty:
        return pd.Series(dtype="float64", name="RankIC_ResidualVol_Monthly")

    valid = factor_df.notna() & future_df.notna()
    valid_count = valid.sum(axis=1)
    factor_rank = factor_df.where(valid).rank(axis=1, method="average")
    future_rank = future_df.where(valid).rank(axis=1, method="average")

    ic_series = factor_rank.corrwith(future_rank, axis=1)
    ic_series = ic_series.where(valid_count >= minimum_stock_count)
    ic_series.name = "RankIC_ResidualVol_Monthly"
    return ic_series.dropna()


def calculate_ic_statistics(ic_series: pd.Series, frequency: int = 12) -> dict[str, float | int]:
    """
    Calculate common IC statistics.

    Metrics include IC mean/std, monthly ICIR, annualized ICIR, t-stat and
    positive-IC hit rate.
    """
    clean_ic = ic_series.dropna()
    count = len(clean_ic)

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

    ic_mean = float(clean_ic.mean())
    ic_std = float(clean_ic.std(ddof=1))

    icir = ic_mean / ic_std if pd.notna(ic_std) and ic_std != 0 else np.nan
    icir_annual = icir * np.sqrt(frequency) if pd.notna(icir) else np.nan
    t_value = ic_mean / (ic_std / np.sqrt(count)) if pd.notna(ic_std) and ic_std != 0 and count > 1 else np.nan
    win_rate = float((clean_ic > 0).mean())

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
# 7) 分组与权重
# ============================================================
def validate_direction(direction: str) -> None:
    """Validate factor direction text."""
    if direction not in {"low_is_good", "high_is_good"}:
        raise ValueError('direction must be either "low_is_good" or "high_is_good"')


def build_group_labels(
    factor_series: pd.Series,
    group_count: int = 5,
    direction: str = "low_is_good",
) -> pd.Series:
    """
    Split stocks into quantile groups by factor value.

    Group1 always contains the lowest factor values and GroupN the highest
    factor values. The direction parameter is kept explicit for interpretation
    and for downstream recommendation sorting.
    """
    validate_direction(direction)
    clean_factor = factor_series.dropna()
    if len(clean_factor) < group_count:
        return pd.Series(dtype="int32")

    ranked_values = clean_factor.rank(method="first")
    group_id = pd.qcut(ranked_values, group_count, labels=False) + 1
    return group_id.astype("int32")


def build_equal_weight_group_portfolio(
    factor_series: pd.Series,
    rebalance_date_str: str,
    group_count: int = 5,
    minimum_stock_count: int = 50,
    direction: str = "low_is_good",
) -> pd.DataFrame:
    """
    Build one rebalance-date equal-weight grouped portfolio table.

    Output columns:
    - 日期
    - 代码
    - 权重
    - 组合名称
    """
    clean_factor = factor_series.dropna()
    if len(clean_factor) < minimum_stock_count:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    group_id_series = build_group_labels(clean_factor, group_count=group_count, direction=direction)
    if group_id_series.empty:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    portfolio_df = pd.DataFrame(
        {
            "日期": rebalance_date_str,
            "代码": group_id_series.index.astype(str),
            "group_id": group_id_series.to_numpy(dtype=np.int32),
        }
    )
    portfolio_df["组合名称"] = "Group" + portfolio_df["group_id"].astype(str)
    portfolio_df["权重"] = 1.0 / portfolio_df.groupby("组合名称")["代码"].transform("size")

    portfolio_df = portfolio_df[["日期", "代码", "权重", "组合名称"]]
    return optimize_portfolio_memory(portfolio_df)


def build_monthly_equal_weight_group_portfolios(
    factor_monthly_df: pd.DataFrame,
    group_count: int = 5,
    minimum_stock_count: int = 50,
    direction: str = "low_is_good",
) -> pd.DataFrame:
    """Build long-format grouped portfolio weights for all rebalance dates."""
    if factor_monthly_df.empty:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    portfolio_list = []
    for month_end_dt, factor_this_month in factor_monthly_df.iterrows():
        portfolio_df = build_equal_weight_group_portfolio(
            factor_series=factor_this_month,
            rebalance_date_str=month_end_dt.strftime("%Y%m%d"),
            group_count=group_count,
            minimum_stock_count=minimum_stock_count,
            direction=direction,
        )
        if not portfolio_df.empty:
            portfolio_list.append(portfolio_df)

    if not portfolio_list:
        return pd.DataFrame(columns=["日期", "代码", "权重", "组合名称"])

    return optimize_portfolio_memory(pd.concat(portfolio_list, ignore_index=True))


# ============================================================
# 8) 月频分组回测
# ============================================================
def backtest_monthly_groups(
    monthly_close_df: pd.DataFrame,
    monthly_portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backtest monthly rebalanced equal-weight groups.

    Weights formed at month t are applied to returns from month t to month t+1.
    The implementation pivots weights by group and uses vectorized dot products
    instead of looping through each date/group/ticker combination.
    """
    if monthly_close_df.empty or monthly_portfolio_df.empty:
        return pd.DataFrame()

    close_df = downcast_float_frame(monthly_close_df).sort_index()
    portfolio_df = monthly_portfolio_df.copy()
    portfolio_df["rebalance_dt"] = pd.to_datetime(portfolio_df["日期"].astype(str), errors="coerce")
    portfolio_df = portfolio_df.dropna(subset=["rebalance_dt"])
    portfolio_df = portfolio_df[portfolio_df["rebalance_dt"].isin(close_df.index)]

    if portfolio_df.empty:
        return pd.DataFrame()

    rebalance_dates = pd.DatetimeIndex(sorted(portfolio_df["rebalance_dt"].unique()))
    if len(rebalance_dates) < 2:
        return pd.DataFrame()

    group_names = sorted(portfolio_df["组合名称"].astype(str).unique().tolist())
    future_return_df = close_df.pct_change(fill_method=None).shift(-1)
    period_index = rebalance_dates[:-1]
    period_return_df = pd.DataFrame(index=period_index, columns=group_names, dtype="float32")

    for group_name in group_names:
        group_holdings = portfolio_df[portfolio_df["组合名称"].astype(str) == group_name]
        if group_holdings.empty:
            continue

        weights = group_holdings.pivot_table(
            index="rebalance_dt",
            columns="代码",
            values="权重",
            aggfunc="sum",
            observed=True,
        )
        weights = weights.reindex(period_index).fillna(0.0)
        aligned_returns = future_return_df.reindex(index=period_index, columns=weights.columns).fillna(0.0)
        period_return_df[group_name] = (weights * aligned_returns).sum(axis=1).astype("float32")

    nav_after_rebalance = (1.0 + period_return_df.fillna(0.0)).cumprod()
    nav_after_rebalance.index = rebalance_dates[1 : len(nav_after_rebalance) + 1]

    initial_nav = pd.DataFrame({group_name: 1.0 for group_name in group_names}, index=[rebalance_dates[0]])
    nav_df = pd.concat([initial_nav, nav_after_rebalance], axis=0)
    nav_df.index.name = "date"
    return downcast_float_frame(nav_df)


# ============================================================
# 9) 赢家组诊断与推荐表
# ============================================================
def diagnose_recent_winner_groups(nav_df: pd.DataFrame, recent_months: int = 12) -> pd.Series:
    """
    Count which group has the highest NAV in the latest N observations.

    Note: this measures the current leading NAV group, not the highest one-month
    return group. It is intended as a simple heuristic for the recommendation step.
    """
    if nav_df.empty or len(nav_df) < 2:
        return pd.Series(dtype="int64")

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
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Build the latest recommendation table from the backtest winner group.

    Inside the selected group:
    - low_is_good: pick the lowest factor values
    - high_is_good: pick the highest factor values
    """
    validate_direction(direction)
    if factor_monthly_df.empty or as_of_month_end not in factor_monthly_df.index:
        return pd.DataFrame()

    factor_today = factor_monthly_df.loc[as_of_month_end].dropna()
    if factor_today.empty:
        return pd.DataFrame()

    group_id_series = build_group_labels(factor_today, group_count=5, direction=direction)
    if group_id_series.empty:
        return pd.DataFrame()

    try:
        target_group_id = int(str(winner_group).replace("Group", ""))
    except ValueError:
        return pd.DataFrame()

    winner_group_tickers = group_id_series.index[group_id_series == target_group_id].tolist()
    if not winner_group_tickers:
        return pd.DataFrame()

    sort_ascending = direction == "low_is_good"
    candidate_factor = factor_today.reindex(winner_group_tickers).dropna()
    candidate_factor = candidate_factor.sort_values(ascending=sort_ascending).head(top_n)
    selected_tickers = candidate_factor.index.tolist()
    if not selected_tickers:
        return pd.DataFrame()

    ret_3m_series = (monthly_close_df / monthly_close_df.shift(3) - 1.0).loc[as_of_month_end]
    ret_12m_series = (monthly_close_df / monthly_close_df.shift(12) - 1.0).loc[as_of_month_end]
    daily_ret_df = daily_close_df.pct_change(fill_method=None)
    ann_vol_series = daily_ret_df.tail(252).std(ddof=1) * np.sqrt(252)
    equal_weight = np.float32(1.0 / len(selected_tickers))

    report_df = pd.DataFrame(
        {
            "as_of_month_end": [as_of_month_end] * len(selected_tickers),
            "winner_group_from_backtest": [winner_group] * len(selected_tickers),
            "ticker": selected_tickers,
            "group": [winner_group] * len(selected_tickers),
            "weight_eq": [equal_weight] * len(selected_tickers),
            "residual_vol": candidate_factor.to_numpy(dtype=np.float32),
            "ret_3m": ret_3m_series.reindex(selected_tickers).to_numpy(dtype=np.float32),
            "ret_12m": ret_12m_series.reindex(selected_tickers).to_numpy(dtype=np.float32),
            "ann_vol_12m": ann_vol_series.reindex(selected_tickers).to_numpy(dtype=np.float32),
            "rule": [f"Winner group = {winner_group} -> pick top_n within group"] * len(selected_tickers),
        }
    )

    for col in ["winner_group_from_backtest", "ticker", "group"]:
        report_df[col] = report_df[col].astype("category")
    return report_df


# ============================================================
# 10) 画图
# ============================================================
def plot_monthly_ic_report(ic_series: pd.Series, title_prefix: str = "Monthly Rank IC") -> None:
    """Plot IC bar chart, IC line chart and cumulative IC chart."""
    ic_series = ic_series.dropna().sort_index()
    if ic_series.empty:
        return

    cumulative_ic = ic_series.cumsum()

    plt.figure()
    plt.bar(ic_series.index, ic_series.values, width=20)
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Bar)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ic_series.index, ic_series.values, marker="o")
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Line)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(cumulative_ic.index, cumulative_ic.values, marker="o")
    plt.axhline(0, linestyle="--")
    plt.title(title_prefix + " (Cumulative)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_group_nav(nav_df: pd.DataFrame, title: str = "Group NAV (Monthly Rebalance)") -> None:
    """Plot grouped portfolio NAV curves."""
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


def plot_long_short_nav(nav_df: pd.DataFrame, long_group: str = "Group5", short_group: str = "Group1") -> None:
    """Plot long-short NAV as long_group / short_group."""
    if nav_df.empty or long_group not in nav_df.columns or short_group not in nav_df.columns:
        return

    long_short_nav = nav_df[long_group] / nav_df[short_group]

    plt.figure()
    plt.plot(long_short_nav.index, long_short_nav.values)
    plt.title(f"Long-Short NAV ({long_group} / {short_group})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================
# 11) 流程编排
# ============================================================
def run_residual_volatility_backtest(
    ticker_list: Sequence[str],
    data_config: DataDownloadConfig | None = None,
    backtest_config: ResidualVolatilityBacktestConfig | None = None,
    plot: bool = True,
    save_outputs: bool = False,
    output_dir: str | Path = "outputs",
) -> BacktestResult | None:
    """
    Run the full residual volatility single-factor research workflow.

    Steps:
    1. Download/cache stock and benchmark close prices
    2. Filter stock pool
    3. Compute monthly residual volatility factor
    4. Calculate Rank IC
    5. Build grouped monthly portfolios and NAV
    6. Produce latest recommendation table
    """
    data_config = data_config or DataDownloadConfig()
    backtest_config = backtest_config or ResidualVolatilityBacktestConfig()
    validate_direction(backtest_config.direction)

    print_divider("下载美股日线 Close")
    print("Ticker count:", len(ticker_list))
    data_source = USStockData(data_config)
    close_price_df, failed_tickers = data_source.get_close_price_matrix(
        ticker_list,
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
    )

    if failed_tickers:
        print("无数据/失败 ticker:", failed_tickers)
    if close_price_df.empty:
        print("没有拉到任何价格数据，结束。")
        return None

    print(f"价格矩阵: {close_price_df.shape}, memory={memory_usage_mb(close_price_df):.2f} MB")

    print_divider("下载市场基准 Close")
    market_close_df, failed_market = data_source.get_close_price_matrix(
        [backtest_config.benchmark_ticker],
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
    )
    if failed_market:
        print("基准下载失败 ticker:", failed_market)
    if market_close_df.empty:
        print(f"{backtest_config.benchmark_ticker} 数据下载失败，无法计算 Residual Volatility。")
        return None

    market_close_series = market_close_df[backtest_config.benchmark_ticker].dropna()

    print_divider("股票池过滤")
    filter_rule_text = (
        f"历史>={backtest_config.minimum_trading_days}天；"
        f"年化波动<={backtest_config.max_annual_volatility:.2f}；"
        f"{backtest_config.min_price:.1f}<=最新价<={backtest_config.max_price:.1f}"
    )
    print("规则:", filter_rule_text)

    kept_tickers, filter_summary_df = filter_stock_pool(
        close_price_df,
        minimum_trading_days=backtest_config.minimum_trading_days,
        min_price=backtest_config.min_price,
        max_price=backtest_config.max_price,
        max_annual_volatility=backtest_config.max_annual_volatility,
    )

    print(f"原始ticker数: {len(close_price_df.columns)}")
    print(f"保留ticker数: {len(kept_tickers)}")
    print(f"剔除ticker数: {len(close_price_df.columns) - len(kept_tickers)}")

    removed_df = filter_summary_df[filter_summary_df["reason"] != ""]
    if not removed_df.empty:
        print("\n--- 剔除清单（前25条）---")
        print(removed_df.head(25)[["ticker", "reason", "trading_days", "last_price", "ann_vol_12m"]].to_string(index=False))

    if len(kept_tickers) < 10:
        print("过滤后可用ticker太少（<10），建议放宽阈值或扩大股票池。")
        return None

    filtered_close_df = downcast_float_frame(close_price_df[kept_tickers].copy())

    print_divider("计算 Residual Volatility 因子")
    factor_monthly_df = build_monthly_residual_vol_factor(
        daily_close_df=filtered_close_df,
        market_daily_close=market_close_series,
        lookback_days=backtest_config.lookback_days,
        minimum_observations=backtest_config.minimum_regression_observations,
    )

    if factor_monthly_df.empty:
        print("Residual Volatility 因子为空，可能是市场数据对齐失败或历史太短。")
        return None

    monthly_close_df = compute_month_end_prices(filtered_close_df)
    common_month_end_dates = factor_monthly_df.index.intersection(monthly_close_df.index)
    factor_monthly_df = factor_monthly_df.loc[common_month_end_dates]
    monthly_close_df = monthly_close_df.loc[common_month_end_dates]

    future_1m_return_df = monthly_close_df.pct_change(fill_method=None).shift(-1)
    ic_series = calculate_monthly_rank_ic(
        factor_monthly_df=factor_monthly_df,
        future_return_df=future_1m_return_df,
        minimum_stock_count=backtest_config.minimum_ic_stock_count,
    )

    print_divider("IC 统计")
    ic_statistics = calculate_ic_statistics(ic_series, frequency=12)
    for key, value in ic_statistics.items():
        if isinstance(value, float):
            if "胜率" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if plot:
        plot_monthly_ic_report(ic_series, title_prefix="Monthly Rank IC (Residual Volatility Factor)")

    monthly_portfolio_df = build_monthly_equal_weight_group_portfolios(
        factor_monthly_df=factor_monthly_df,
        group_count=backtest_config.group_count,
        minimum_stock_count=backtest_config.minimum_group_stock_count,
        direction=backtest_config.direction,
    )

    if monthly_portfolio_df.empty:
        print("portfolio 为空：可用股票太少或因子缺失太多。")
        return None

    nav_df = backtest_monthly_groups(
        monthly_close_df=monthly_close_df,
        monthly_portfolio_df=monthly_portfolio_df,
    )

    print_divider("净值（最后5行）")
    if nav_df.empty:
        print("Empty NAV dataframe.")
        return None
    print(nav_df.tail())

    if plot:
        plot_group_nav(nav_df, title="Group NAV (Residual Volatility Factor, Monthly Rebalance)")
        plot_long_short_nav(nav_df)

    winner_count = diagnose_recent_winner_groups(nav_df, recent_months=backtest_config.recent_winner_months)
    print_divider(f"诊断：最近赢家组频次（近{backtest_config.recent_winner_months}个月）")
    if winner_count.empty:
        print("无法统计赢家组，nav_df 为空或月份不足。")
        return None
    print(winner_count.to_string())

    winner_group = str(winner_count.index[0])
    latest_month_end = factor_monthly_df.index.max()
    recommendation_df = build_recommendation_table(
        as_of_month_end=latest_month_end,
        winner_group=winner_group,
        factor_monthly_df=factor_monthly_df,
        monthly_close_df=monthly_close_df,
        daily_close_df=filtered_close_df,
        direction=backtest_config.direction,
        top_n=backtest_config.recommendation_top_n,
    )

    print_divider("最新一期推荐（赢家组）")
    if recommendation_df.empty:
        print("推荐表为空：可能赢家组当期没有足够股票或因子缺失。")
    else:
        print(f"调仓月末: {latest_month_end.date()} | 赢家组: {winner_group} | 等权持仓数: {len(recommendation_df)}")
        print("Tickers:", recommendation_df["ticker"].astype(str).tolist())

        print_divider("推荐表（基于：回测赢家组 + 最新一期因子分组）")
        display_df = recommendation_df.copy()
        display_df["weight_eq"] = display_df["weight_eq"].apply(lambda x: format_percent(x, 2))
        display_df["ret_3m"] = display_df["ret_3m"].apply(lambda x: format_percent(x, 2))
        display_df["ret_12m"] = display_df["ret_12m"].apply(lambda x: format_percent(x, 2))
        display_df["ann_vol_12m"] = display_df["ann_vol_12m"].apply(lambda x: format_percent(x, 2))
        print(display_df.to_string(index=False))

    result = BacktestResult(
        close_price_df=close_price_df,
        filtered_close_df=filtered_close_df,
        market_close_series=market_close_series,
        filter_summary_df=filter_summary_df,
        factor_monthly_df=factor_monthly_df,
        ic_series=ic_series,
        ic_statistics=ic_statistics,
        monthly_portfolio_df=monthly_portfolio_df,
        nav_df=nav_df,
        recommendation_df=recommendation_df,
    )

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ic_series.to_csv(output_path / "ic_series.csv", encoding="utf-8-sig")
        nav_df.to_csv(output_path / "group_nav.csv", encoding="utf-8-sig")
        recommendation_df.to_csv(output_path / "recommendation_table.csv", index=False, encoding="utf-8-sig")
        filter_summary_df.to_csv(output_path / "filter_summary.csv", index=False, encoding="utf-8-sig")
        print(f"已保存结果到: {output_path.resolve()}")

    return result


# ============================================================
# 12) main：示例股票池
# ============================================================
def main() -> None:
    """Run the residual volatility backtest with the default ticker pool."""
    base_ticker_list = [
        "GE", "GEV", "ROBT", "AIQ", "CGNX", "TER", "MU", "TSM", "AMD", "AAPL", "INTC",
        "BMNR", "ISRG", "IAU", "NVDA", "TSLA", "NTD0Y", "PLTR", "SONY", "PFE", "DJT",
        "AMZN", "MSTR", "COIN", "BURBY", "JANX", "ASST", "AMZE", "IBRX", "SLGB", "GITS",
        "DVLT", "NU", "GORO", "GLD",
        "SPLG", "QCOM", "TXN", "AMAT", "ADI", "CSCO", "DELL", "HPQ", "UBER",
        "BAC", "WFC", "C", "SCHW", "COF", "USB",
        "MRK", "BMY", "GILD", "CVS",
        "KO", "TGT", "NKE", "SBUX", "DIS", "EBAY",
        "OXY", "SLB", "COP", "VLO",
        "DUK", "SO", "EXC", "STX",
        "O", "VICI", "WDC", "NTAP",
    ]

    data_config = DataDownloadConfig(
        cache_dir=Path("data_cache") / "us_stock",
        use_cache=True,
        force_refresh=False,
        cache_stale_days=1,
        max_retries=3,
        retry_sleep_seconds=1.5,
        max_workers=4,
        verbose=True,
    )
    backtest_config = ResidualVolatilityBacktestConfig(
        benchmark_ticker="SPY",
        minimum_trading_days=504,
        min_price=5.0,
        max_price=150.0,
        max_annual_volatility=0.80,
        lookback_days=63,
        minimum_regression_observations=30,
        group_count=5,
        direction="low_is_good",
        minimum_ic_stock_count=20,
        minimum_group_stock_count=20,
        recommendation_top_n=10,
        recent_winner_months=12,
    )

    run_residual_volatility_backtest(
        ticker_list=base_ticker_list,
        data_config=data_config,
        backtest_config=backtest_config,
        plot=True,
        save_outputs=False,
    )


if __name__ == "__main__":
    main()
