"""Residual-volatility single-factor research workflow for US equities.
美股残差波动率单因子研究流程。

The script downloads daily US stock data through AkShare, builds a CAPM
idiosyncratic-volatility factor, validates it with monthly Rank IC, and runs an
equal-weight grouped backtest.
本脚本通过 AkShare 下载美股日频数据，构建 CAPM 特异波动率因子，使用月度
Rank IC 做有效性检验，并进行等权分组回测。

The code favors explicit financial assumptions over hidden helper magic so that
research choices remain easy to audit.
代码刻意保留清晰的金融假设，便于后续复核每一个研究选择。
"""

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


# General utilities / 通用工具


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert date-like values to pandas datetime.
    将类日期数据转换为 pandas datetime。

    Args:
        series (pd.Series): Source date-like values. 原始类日期序列。

    Returns:
        pd.Series: Datetime series where invalid values are coerced to ``NaT``.
        转换后的日期序列；非法日期会被置为 ``NaT``。
    """
    return pd.to_datetime(series, errors="coerce")


def format_percent(value: float, digits: int = 2) -> str:
    """Format a decimal number as percentage text.
    将小数格式化为百分比文本。

    Args:
        value (float): Decimal return, weight, or ratio. 小数形式的收益、权重或比率。
        digits (int): Number of decimal places to display. 保留的小数位数。

    Returns:
        str: Percentage text, or ``"NaN"`` when ``value`` is missing.
        百分比字符串；缺失值返回 ``"NaN"``。
    """
    if pd.isna(value):
        return "NaN"
    return f"{value * 100:.{digits}f}%"


def annualized_volatility(daily_return_series: pd.Series, trading_days: int = 252) -> float:
    """Calculate annualized volatility from daily returns.
    基于日收益率计算年化波动率。

    Args:
        daily_return_series (pd.Series): Daily return observations.
            日收益率序列。
        trading_days (int): Annualization base. US equity research commonly
            uses 252 trading days.
            年化交易日数量；美股研究通常使用 252 个交易日。

    Returns:
        float: Annualized volatility, or ``np.nan`` when no valid returns exist.
        年化波动率；若没有有效收益率，则返回 ``np.nan``。
    """
    clean_return = daily_return_series.dropna()
    if clean_return.empty:
        return np.nan
    return float(clean_return.std(ddof=1) * np.sqrt(trading_days))


def print_divider(title: str) -> None:
    """Print a readable console section divider.
    打印可读的控制台分区标题。

    Args:
        title (str): Section title shown in the console. 控制台分区标题。

    Returns:
        None. 无返回值。
    """
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def memory_usage_mb(frame: pd.DataFrame) -> float:
    """Measure deep DataFrame memory usage in megabytes.
    以 MB 为单位统计 DataFrame 的深度内存占用。

    Args:
        frame (pd.DataFrame): DataFrame to inspect. 待检查的 DataFrame。

    Returns:
        float: Deep memory usage in MB. 深度内存占用，单位 MB。
    """
    if frame.empty:
        return 0.0
    return float(frame.memory_usage(deep=True).sum() / 1024**2)


def downcast_float_frame(frame: pd.DataFrame, dtype: str = "float32") -> pd.DataFrame:
    """Downcast numeric DataFrame values to reduce memory usage.
    将数值型 DataFrame 向下转换，以降低内存占用。

    float32 is usually enough for daily prices, returns and factor values in this
    script. If you need very high precision accounting numbers, pass dtype="float64".
    对日线价格、收益率和因子值而言，``float32`` 通常已经足够；若处理高精度
    会计数据，可以传入 ``dtype="float64"``。

    Args:
        frame (pd.DataFrame): Numeric-like input DataFrame. 类数值型输入矩阵。
        dtype (str): Target floating dtype, typically ``"float32"``.
            目标浮点类型，通常为 ``"float32"``。

    Returns:
        pd.DataFrame: Numeric DataFrame converted to the target dtype.
        转换到目标类型后的数值矩阵。
    """
    if frame.empty:
        return frame
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    return numeric_frame.astype(dtype, copy=False)


def optimize_portfolio_memory(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Use compact dtypes for the long-format portfolio table.
    为长表形式的组合权重表使用更紧凑的数据类型。

    Args:
        portfolio_df (pd.DataFrame): Portfolio table with ticker, group, and
            weight columns.
            包含 ticker、分组和权重字段的组合表。

    Returns:
        pd.DataFrame: Copy of the portfolio table with compact dtypes.
        使用紧凑类型后的组合表副本。
    """
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
    """Resample daily data to month-end using the last available observation.
    使用每月最后一个可用观测值，将日频数据压缩为月末数据。

    Pandas 2.2 prefers "ME"; older versions may only support "M", so this keeps
    the script compatible across common research environments.
    Pandas 2.2 推荐使用 ``"ME"``；较旧版本可能只支持 ``"M"``，这里做兼容处理。

    Args:
        data (pd.DataFrame | pd.Series): Daily time series with a DatetimeIndex.
            以 DatetimeIndex 为索引的日频序列或矩阵。

    Returns:
        pd.DataFrame | pd.Series: Month-end series or matrix. 月末序列或矩阵。
    """
    try:
        return data.resample("ME").last()
    except ValueError:
        return data.resample("M").last()


# Configuration objects / 配置对象


@dataclass(frozen=True)
class DataDownloadConfig:
    """Configuration for AkShare downloads and local cache behavior.
    AkShare 下载与本地缓存配置。

    Args:
        cache_dir (str | Path): Directory used to persist per-ticker close
            price caches.
            每个 ticker 的收盘价缓存目录。
        use_cache (bool): Whether local cache is enabled. 是否启用本地缓存。
        force_refresh (bool): Whether to ignore fresh cache and request data
            again from AkShare.
            是否忽略新鲜缓存并强制重新请求 AkShare。
        cache_stale_days (int | None): Maximum cache age in calendar days.
            ``None`` means cached files never expire.
            缓存最大有效天数；``None`` 表示缓存永不过期。
        max_retries (int): Maximum attempts for each AkShare request.
            每个 AkShare 请求的最大重试次数。
        retry_sleep_seconds (float): Base delay between retries. Later retries
            use exponential backoff.
            重试基础等待秒数；后续重试使用指数退避。
        max_workers (int): Maximum concurrent ticker downloads. 并发下载线程数上限。
        verbose (bool): Whether to print download/cache diagnostics.
            是否打印下载和缓存诊断信息。

    Returns:
        DataDownloadConfig: Immutable download/cache configuration instance.
        不可变的下载与缓存配置实例。
    """

    cache_dir: str | Path = field(
        default_factory=lambda: Path("data_cache") / "us_stock"
    )
    use_cache: bool = True
    force_refresh: bool = False
    cache_stale_days: int | None = 1
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5
    max_workers: int = 4
    verbose: bool = True


@dataclass(frozen=True)
class ResidualVolatilityBacktestConfig:
    """Configuration for the residual-volatility backtest workflow.
    残差波动率回测流程配置。

    Args:
        start_date (str | None): Optional inclusive start date for all series.
            所有序列的可选起始日期，闭区间。
        end_date (str | None): Optional inclusive end date for all series.
            所有序列的可选结束日期，闭区间。
        benchmark_ticker (str): Market proxy used in CAPM regression.
            CAPM 回归使用的市场代理 ticker。
        minimum_trading_days (int): Minimum valid price observations required
            before a stock enters the research universe.
            股票进入研究池前所需的最少有效价格观测数。
        min_price (float): Minimum latest close price filter. 最新价下限。
        max_price (float): Maximum latest close price filter. 最新价上限。
        max_annual_volatility (float): Maximum trailing 12-month annualized
            volatility allowed in the universe.
            股票池允许的最大近 12 个月年化波动率。
        lookback_days (int): Rolling regression window length in trading days.
            The default 63 days is roughly one quarter, a practical compromise
            between fast regime response and enough observations for beta.
            滚动回归窗口长度；默认 63 个交易日约等于一个季度，兼顾 beta 估计
            的样本数和风险状态变化的响应速度。
        minimum_regression_observations (int): Minimum paired stock/market
            returns required inside a rolling CAPM window.
            CAPM 滚动窗口内所需的最少股票/市场配对收益率观测数。
        group_count (int): Number of factor quantile groups. 因子分组数量。
        direction (str): Factor interpretation, either ``"low_is_good"`` or
            ``"high_is_good"``.
            因子方向解释，可为 ``"low_is_good"`` 或 ``"high_is_good"``。
        minimum_ic_stock_count (int): Minimum cross-sectional sample size for
            each monthly Rank IC.
            每个月 Rank IC 所需的最小横截面股票数量。
        minimum_group_stock_count (int): Minimum valid factor count before
            building grouped portfolios at one rebalance date.
            单个调仓日构建分组组合前所需的最小有效因子数量。
        signal_lag_periods (int): Number of monthly periods between signal
            observation and return application.
            信号观察和收益应用之间滞后的月度期数。
        transaction_cost_bps (float): One-way transaction cost in basis points
            applied to absolute turnover.
            按绝对换手率扣减的单边交易成本，单位为基点。
        recommendation_top_n (int): Number of names shown in the final
            recommendation table.
            最终推荐表展示的股票数量。
        recent_winner_months (int): Number of recent observations used to
            identify the simple winner-group heuristic.
            用于识别近期赢家组的最近观测期数。

    Returns:
        ResidualVolatilityBacktestConfig: Immutable workflow configuration
        instance.
        不可变的回测流程配置实例。
    """

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
    signal_lag_periods: int = 1
    transaction_cost_bps: float = 10.0
    recommendation_top_n: int = 10
    recent_winner_months: int = 12


@dataclass
class MonthlyGroupBacktestResult:
    """Container for monthly group backtest diagnostics.
    月度分组回测诊断结果容器。

    Args:
        nav_df (pd.DataFrame): Net group NAV after transaction costs.
            扣除交易成本后的分组净值矩阵。
        gross_nav_df (pd.DataFrame): Gross group NAV before transaction costs.
            扣除交易成本前的分组毛净值矩阵。
        period_return_df (pd.DataFrame): Net period returns after costs.
            扣除交易成本后的单期收益率矩阵。
        gross_return_df (pd.DataFrame): Gross period returns before costs.
            扣除交易成本前的单期收益率矩阵。
        turnover_df (pd.DataFrame): Absolute turnover by group and period.
            各分组、各期的绝对换手率矩阵。
        transaction_cost_df (pd.DataFrame): Cost drag by group and period.
            各分组、各期的交易成本拖累矩阵。
        schedule_df (pd.DataFrame): Mapping between holding-period start/end
            dates and signal dates.
            持有期起止日期与信号日期的映射表。

    Returns:
        MonthlyGroupBacktestResult: Structured monthly group backtest result.
        结构化的月度分组回测结果。
    """

    nav_df: pd.DataFrame
    gross_nav_df: pd.DataFrame
    period_return_df: pd.DataFrame
    gross_return_df: pd.DataFrame
    turnover_df: pd.DataFrame
    transaction_cost_df: pd.DataFrame
    schedule_df: pd.DataFrame


@dataclass
class BacktestResult:
    """Container for major intermediate and final backtest outputs.
    回测主要中间结果和最终结果容器。

    Args:
        close_price_df (pd.DataFrame): Raw aligned close-price matrix.
            原始对齐后的收盘价矩阵。
        filtered_close_df (pd.DataFrame): Close-price matrix after universe
            filters.
            股票池过滤后的收盘价矩阵。
        market_close_series (pd.Series): Benchmark close-price series.
            基准收盘价序列。
        filter_summary_df (pd.DataFrame): Per-ticker universe filter report.
            逐 ticker 股票池过滤报告。
        factor_monthly_df (pd.DataFrame): Monthly residual-volatility factor.
            月度残差波动率因子矩阵。
        ic_series (pd.Series): Monthly Rank IC series. 月度 Rank IC 序列。
        ic_statistics (dict[str, float | int]): Summary IC diagnostics.
            IC 汇总诊断指标。
        monthly_portfolio_df (pd.DataFrame): Long-format grouped weights.
            长表形式的月度分组权重。
        nav_df (pd.DataFrame): Group NAV matrix. 分组净值矩阵。
        gross_nav_df (pd.DataFrame): Group NAV matrix before transaction costs.
            扣除交易成本前的分组毛净值矩阵。
        turnover_df (pd.DataFrame): Group turnover matrix. 分组换手率矩阵。
        transaction_cost_df (pd.DataFrame): Group transaction-cost matrix.
            分组交易成本矩阵。
        backtest_schedule_df (pd.DataFrame): Signal date and holding-period
            date mapping used by the grouped backtest.
            分组回测使用的信号日期与持有期日期映射表。
        recommendation_df (pd.DataFrame): Latest recommendation table.
            最新一期推荐表。

    Returns:
        BacktestResult: Structured output container for downstream analysis.
        供后续分析使用的结构化结果容器。
    """

    close_price_df: pd.DataFrame
    filtered_close_df: pd.DataFrame
    market_close_series: pd.Series
    filter_summary_df: pd.DataFrame
    factor_monthly_df: pd.DataFrame
    ic_series: pd.Series
    ic_statistics: dict[str, float | int]
    monthly_portfolio_df: pd.DataFrame
    nav_df: pd.DataFrame
    gross_nav_df: pd.DataFrame
    turnover_df: pd.DataFrame
    transaction_cost_df: pd.DataFrame
    backtest_schedule_df: pd.DataFrame
    recommendation_df: pd.DataFrame


# AkShare data access / AkShare 数据访问


class USStockData:
    """Download US stock close prices from AkShare with retry and cache support.
    通过 AkShare 下载美股收盘价，并提供重试和缓存支持。

    The class caches each ticker separately so a failed request does not force a
    full-universe redownload. In research workflows, stale-but-recent data is
    often more useful than failing the whole run when one AkShare endpoint times
    out.
    本类按 ticker 单独缓存，因此单个请求失败不会迫使整个股票池重新下载。
    对研究流程来说，使用较新的旧缓存通常比因 AkShare 短暂超时而中断整个流程更实用。

    Args:
        config (DataDownloadConfig | None): Download and cache settings. If
            omitted, a default ``DataDownloadConfig`` is created.
            下载和缓存配置；若省略，则使用默认配置。

    Returns:
        USStockData: Data access object that can download aligned close-price
        matrices.
        可下载并对齐收盘价矩阵的数据访问对象。

    Attributes:
        config (DataDownloadConfig): Download and cache settings.
            下载与缓存配置。
        cache_dir (Path): Resolved local cache directory. 解析后的本地缓存目录。
    """

    def __init__(self, config: DataDownloadConfig | None = None) -> None:
        """Initialize the data source.
        初始化数据源对象。

        Args:
            config (DataDownloadConfig | None): Optional download/cache
                configuration. Defaults are used when omitted.
                可选下载/缓存配置；若省略则使用默认配置。

        Returns:
            None. 无返回值。
        """
        self.config = config or DataDownloadConfig()
        self.cache_dir = Path(self.config.cache_dir)

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:
        """Normalize ticker text for AkShare calls and cache file names.
        标准化 ticker 文本，用于 AkShare 请求和缓存文件名。

        Args:
            ticker (str): Raw ticker text. 原始 ticker 文本。

        Returns:
            str: Uppercase ticker with surrounding whitespace removed.
            去除首尾空格并转换为大写后的 ticker。
        """
        return str(ticker).strip().upper()

    def _cache_path(self, ticker: str) -> Path:
        """Build the local cache path for one ticker.
        构建单个 ticker 的本地缓存路径。

        Args:
            ticker (str): Stock ticker. 股票 ticker。

        Returns:
            Path: Per-ticker pickle cache path. 单 ticker 的 pickle 缓存路径。
        """
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._sanitize_ticker(ticker))
        return self.cache_dir / f"{safe_name}.pkl"

    def _cache_is_fresh(self, ticker: str) -> bool:
        """Check whether a ticker cache exists and is still fresh.
        检查单个 ticker 的缓存是否存在且仍然有效。

        Args:
            ticker (str): Stock ticker. 股票 ticker。

        Returns:
            bool: ``True`` when cache exists and does not exceed the configured
            age threshold.
            若缓存存在且未超过配置的有效期，则返回 ``True``。
        """
        path = self._cache_path(ticker)
        if not path.exists():
            return False
        if self.config.cache_stale_days is None:
            return True
        max_age_seconds = self.config.cache_stale_days * 24 * 60 * 60
        return (time.time() - path.stat().st_mtime) <= max_age_seconds

    def _read_cached_ticker_close(self, ticker: str) -> pd.Series | None:
        """Read a cached close-price series if available and valid.
        读取可用且有效的本地收盘价缓存。

        Args:
            ticker (str): Stock ticker. 股票 ticker。

        Returns:
            pd.Series | None: Cached close series indexed by date, or ``None``
            when the cache is missing/corrupted.
            以日期为索引的缓存收盘价序列；若缓存缺失或损坏则返回 ``None``。
        """
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
        """Persist a ticker close series to local pickle cache.
        将单个 ticker 的收盘价序列落盘为本地 pickle 缓存。

        Args:
            ticker (str): Stock ticker. 股票 ticker。
            series (pd.Series): Close-price series indexed by date.
                以日期为索引的收盘价序列。

        Returns:
            None. 无返回值。
        """
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
        """Normalize different AkShare schemas into one close-price series.
        将 AkShare 不同接口返回的字段结构统一为收盘价序列。

        Args:
            df (pd.DataFrame): Raw DataFrame returned by an AkShare endpoint.
                AkShare 接口返回的原始 DataFrame。
            ticker (str): Stock ticker used as the output series name.
                输出序列名称使用的股票 ticker。

        Returns:
            pd.Series | None: Clean close-price series indexed by date, or
            ``None`` when the schema has no recognizable date/close columns.
            清洗后的收盘价序列；若无法识别日期/收盘价字段，则返回 ``None``。
        """
        if df is None or df.empty:
            return None

        lower_to_original = {str(col).lower(): col for col in df.columns}
        date_candidates = ["date", "日期", "Date", "时间", "datetime"]
        close_candidates = ["close", "收盘", "Close", "adj_close", "adjusted_close"]

        date_col = next((col for col in date_candidates if col in df.columns), None)
        if date_col is None:
            date_col = next(
                (
                    lower_to_original[col.lower()]
                    for col in date_candidates
                    if col.lower() in lower_to_original
                ),
                None,
            )

        close_col = next((col for col in close_candidates if col in df.columns), None)
        if close_col is None:
            close_col = next(
                (
                    lower_to_original[col.lower()]
                    for col in close_candidates
                    if col.lower() in lower_to_original
                ),
                None,
            )

        if date_col is None or close_col is None:
            return None

        normalized_df = df[[date_col, close_col]].copy()
        normalized_df[date_col] = safe_to_datetime(normalized_df[date_col])
        normalized_df[close_col] = pd.to_numeric(normalized_df[close_col], errors="coerce")
        normalized_df = normalized_df.dropna(subset=[date_col, close_col])
        if normalized_df.empty:
            return None

        series = (
            normalized_df.drop_duplicates(subset=[date_col], keep="last")
            .set_index(date_col)[close_col]
        )
        series = series.sort_index()
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_convert(None)

        return series.astype("float32").rename(ticker)

    def _fetch_from_akshare(self, ticker: str) -> pd.Series | None:
        """Fetch one ticker from AkShare.
        从 AkShare 获取单个 ticker 的收盘价数据。

        AkShare has used different functions/schemas across versions, so this
        method tries the two common US equity endpoints and normalizes the result.
        AkShare 不同版本可能暴露不同函数或字段结构，因此这里尝试两个常见美股接口，
        并将结果统一清洗。

        Args:
            ticker (str): Stock ticker accepted by AkShare. AkShare 接受的股票 ticker。

        Returns:
            pd.Series | None: Close-price series indexed by date. 以日期为索引的收盘价序列。

        Raises:
            RuntimeError: Raised when all supported AkShare endpoints fail or
                return unsupported schemas.
            当所有支持的 AkShare 接口均失败或字段结构不支持时抛出。
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
        """Download one ticker close series with cache, retry, and fallback.
        下载单个 ticker 的收盘价序列，包含缓存、重试和兜底逻辑。

        Fresh cache is used first because repeated full-universe downloads are
        slow and fragile. If fresh cache is unavailable, the function retries
        AkShare requests; if those fail, a stale cache is returned when present
        so that one network outage does not invalidate the whole research run.
        优先使用新鲜缓存，是因为全股票池重复下载既慢又容易受网络波动影响。
        若无新鲜缓存，则重试 AkShare；若仍失败且存在旧缓存，则返回旧缓存，避免
        单次网络异常导致整次研究流程失效。

    Args:
            ticker (str): Stock ticker. 股票 ticker。

        Returns:
            pd.Series | None: Close-price series indexed by date, or ``None``
            when neither AkShare nor cache provides usable data.
            以日期为索引的收盘价序列；若 AkShare 和缓存都没有可用数据，则返回 ``None``。
        """
        ticker = self._sanitize_ticker(ticker)
        cached_series = (
            self._read_cached_ticker_close(ticker)
            if self.config.use_cache
            else None
        )

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
                # Exponential backoff is intentionally conservative: AkShare
                # endpoints may rate-limit bursty requests during market hours.
                # 指数退避故意偏保守：AkShare 接口在交易时段可能对突发请求限流。
                sleep_seconds = self.config.retry_sleep_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

        if cached_series is not None:
            if self.config.verbose:
                print(
                    f"{ticker} download failed, using stale cache. "
                    f"Last error: {repr(last_error)}"
                )
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
        """Clip a price series by optional inclusive dates.
        按可选的闭区间日期裁剪价格序列。

        Args:
            series (pd.Series): Price series indexed by date. 以日期为索引的价格序列。
            start_date (str | None): Inclusive start date. 起始日期，闭区间。
            end_date (str | None): Inclusive end date. 结束日期，闭区间。

        Returns:
            pd.Series: Date-filtered price series. 日期过滤后的价格序列。
        """
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
        """Download and align ticker close prices.
        下载并对齐多个 ticker 的收盘价。

        Args:
            ticker_list (Sequence[str]): Stock tickers to download. 待下载的股票 ticker 列表。
            start_date (str | None): Optional inclusive start date. 可选起始日期，闭区间。
            end_date (str | None): Optional inclusive end date. 可选结束日期，闭区间。

        Returns:
            tuple[pd.DataFrame, list[str]]: ``close_price_df`` is indexed by
            trading date with ticker columns and close-price values;
            ``failed_tickers`` contains tickers without usable data after
            download/cache/date filtering.
            返回 ``close_price_df`` 和 ``failed_tickers``；前者为日期索引、ticker
            列的收盘价矩阵，后者为下载/缓存/日期过滤后仍无可用数据的 ticker 列表。
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
            """Load and date-filter one ticker.
            加载并日期过滤单个 ticker。

            Args:
                ticker (str): Normalized stock ticker. 标准化后的股票 ticker。

            Returns:
                tuple[str, pd.Series | None]: Ticker and its filtered close
                series, or ``None`` when no usable data exists.
                ticker 及其过滤后的收盘价序列；若无可用数据则序列为 ``None``。
            """
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
                future_map = {
                    executor.submit(load_one, ticker): ticker
                    for ticker in normalized_tickers
                }
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
            [
                close_by_ticker[ticker]
                for ticker in normalized_tickers
                if ticker in close_by_ticker
            ],
            axis=1,
        ).sort_index()
        close_price_df = close_price_df[~close_price_df.index.duplicated(keep="last")]
        close_price_df = close_price_df.dropna(how="all").ffill()
        close_price_df = close_price_df.reindex(
            columns=[
                ticker
                for ticker in normalized_tickers
                if ticker in close_price_df.columns
            ]
        )

        return downcast_float_frame(close_price_df), failed_tickers


# Universe filtering / 股票池过滤


def filter_stock_pool(
    close_price_df: pd.DataFrame,
    minimum_trading_days: int = 504,
    min_price: float = 5.0,
    max_price: float = 150.0,
    max_annual_volatility: float = 0.80,
) -> tuple[list[str], pd.DataFrame]:
    """Filter the stock universe by investability and data quality.
    按可投资性和数据质量过滤股票池。

    The numeric statistics are vectorized. Only the human-readable reason string
    uses a light ticker loop, which is not performance critical.
    数值统计部分已向量化；仅用于展示的剔除原因字符串使用轻量循环，不构成性能瓶颈。

    Args:
        close_price_df (pd.DataFrame): Daily close-price matrix indexed by date
            with ticker columns.
            日期索引、ticker 列的日频收盘价矩阵。
        minimum_trading_days (int): Minimum valid close observations required.
            最少有效收盘价观测数。
        min_price (float): Minimum latest close price allowed. 最新收盘价下限。
        max_price (float): Maximum latest close price allowed. 最新收盘价上限。
        max_annual_volatility (float): Maximum trailing one-year annualized
            volatility allowed.
            允许的最大近一年年化波动率。

    Returns:
        tuple[list[str], pd.DataFrame]: ``kept_tickers`` contains tickers that
        pass all filters. ``summary_df`` contains ticker-level diagnostics with
        columns ``ticker``, ``trading_days``, ``last_price``, ``ann_vol_12m``,
        and ``reason``.
        ``kept_tickers`` 为通过过滤的 ticker；``summary_df`` 为逐 ticker 诊断表，
        包含 ``ticker``、``trading_days``、``last_price``、``ann_vol_12m`` 和
        ``reason``。
    """
    if close_price_df.empty:
        return [], pd.DataFrame()

    clean_close_df = downcast_float_frame(close_price_df).sort_index()
    daily_return_df = clean_close_df.pct_change(fill_method=None)

    # The filters are intentionally simple and transparent: penny-like prices,
    # very short histories, and extreme realized volatility can dominate a
    # residual-volatility sort without representing a stable factor premium.
    # 这些过滤规则刻意保持简单透明：低价股、短历史和极端波动会主导残差波动率排序，
    # 但未必代表稳定可交易的因子溢价。
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


# Monthly sampling / 月末采样


def compute_month_end_prices(daily_close_df: pd.DataFrame) -> pd.DataFrame:
    """Compress daily close prices to month-end close prices.
    将日频收盘价压缩为月末收盘价。

    Args:
        daily_close_df (pd.DataFrame): Daily close-price matrix indexed by
            trading date.
            以交易日为索引的日频收盘价矩阵。

    Returns:
        pd.DataFrame: Month-end close-price matrix. 月末收盘价矩阵。
    """
    if daily_close_df.empty:
        return pd.DataFrame()
    monthly_close_df = resample_month_end_last(daily_close_df).dropna(how="all")
    return downcast_float_frame(monthly_close_df)


# Residual-volatility factor / 残差波动率因子


def compute_capm_residual_volatility(
    stock_daily_return: pd.Series,
    market_daily_return: pd.Series,
) -> float:
    """Compute single-stock CAPM residual volatility.
    计算单只股票的 CAPM 残差波动率。

    This function is kept for diagnostics and unit checks. The production factor
    path uses compute_rolling_capm_residual_volatility_matrix for speed.
    该函数保留给诊断和单元校验使用；正式因子计算走矩阵化函数以提高速度。

    Args:
        stock_daily_return (pd.Series): Daily stock returns. 股票日收益率序列。
        market_daily_return (pd.Series): Daily benchmark returns aligned by
            date where possible.
            尽可能按日期对齐后的市场基准日收益率序列。

    Returns:
        float: Standard deviation of CAPM residuals, or ``np.nan`` when the
        regression window is unusable.
        CAPM 残差标准差；若回归窗口不可用，则返回 ``np.nan``。
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
    """Calculate vectorized rolling CAPM residual volatility for all tickers.
    为所有 ticker 向量化计算滚动 CAPM 残差波动率。

    For each rolling window and ticker, the function estimates:
        r_stock = alpha + beta * r_market + residual
    对每个滚动窗口和 ticker，函数估计：
        r_stock = alpha + beta * r_market + residual

    It avoids the slow month-end x ticker regression loop by using rolling sums:
        Sxx, Syy, Sxy -> beta -> SSE -> residual std
    函数通过滚动求和恢复回归统计量，避免“月末 × ticker”的低效逐个回归：
        Sxx, Syy, Sxy -> beta -> SSE -> residual std

    The output matches the previous convention of std(residual, ddof=1), not the
    classical regression sigma with n-2 degrees of freedom.
    输出沿用原脚本的 ``std(residual, ddof=1)`` 口径，而不是经典回归的 n-2
    自由度残差标准误。

    A 63-trading-day window is the default in the caller because it approximates
    one quarter of US trading days. That is usually enough to estimate a beta
    while still adapting faster than a 6- or 12-month realized-risk measure.
    默认 63 个交易日约等于一个季度：既能提供足够样本估计 beta，又比 6 个月或
    12 个月的风险度量更快反映市场状态变化。

    Args:
        stock_daily_return_df (pd.DataFrame): Daily stock-return matrix indexed
            by date with ticker columns.
            日期索引、ticker 列的股票日收益率矩阵。
        market_daily_return (pd.Series): Daily benchmark returns. 市场基准日收益率序列。
        lookback_days (int): Rolling CAPM window length in trading days.
            滚动 CAPM 窗口长度，单位为交易日。
        minimum_observations (int): Minimum paired stock/market observations
            required in each rolling window.
            每个滚动窗口中所需的最少股票/市场配对观测数。

    Returns:
        pd.DataFrame: Daily residual-volatility matrix indexed by date with
        ticker columns.
        日期索引、ticker 列的日频残差波动率矩阵。

    Raises:
        ValueError: Raised when ``lookback_days`` is smaller than 3.
        当 ``lookback_days`` 小于 3 时抛出。
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

    # Rolling sums let us recover beta and residual variance algebraically. This
    # keeps the financial model unchanged while avoiding thousands of tiny OLS
    # fits across month-end dates and tickers.
    # 滚动求和让我们用代数方式恢复 beta 和残差方差；金融模型不变，但避免了
    # 在大量月末日期和 ticker 上反复执行小规模 OLS。
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
    """Build the monthly residual-volatility factor.
    构建月度残差波动率因子。

    Residual volatility is interpreted as idiosyncratic risk: the part of a
    stock's daily return variation not explained by the market proxy. Sampling
    the daily rolling estimate at month-end makes the factor compatible with a
    monthly rebalance schedule.
    残差波动率可理解为个股特异风险：即无法被市场代理解释的股票日收益波动。
    在月末抽取日频滚动估计值，可以与月度调仓节奏保持一致。

    Args:
        daily_close_df (pd.DataFrame): Daily close-price matrix indexed by date
            with ticker columns.
            日期索引、ticker 列的日频收盘价矩阵。
        market_daily_close (pd.Series): Daily benchmark close prices. 基准日频收盘价。
        lookback_days (int): Rolling CAPM window length in trading days.
            滚动 CAPM 窗口长度，单位为交易日。
        minimum_observations (int): Minimum paired observations required in the
            rolling regression window.
            滚动回归窗口内所需的最少配对观测数。

    Returns:
        pd.DataFrame: Monthly factor matrix indexed by month-end date with
        ticker columns.
        月末日期索引、ticker 列的月度因子矩阵。
    """
    if daily_close_df.empty or market_daily_close.dropna().empty:
        return pd.DataFrame()

    daily_close_df = downcast_float_frame(daily_close_df).sort_index()
    market_daily_close = pd.to_numeric(market_daily_close.sort_index(), errors="coerce")

    stock_daily_return_df = daily_close_df.pct_change(fill_method=None)
    market_daily_return = market_daily_close.pct_change(fill_method=None).reindex(
        stock_daily_return_df.index
    )

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


# Rank IC diagnostics / Rank IC 诊断


def calculate_spearman_rank_ic(
    factor_series: pd.Series,
    future_return_series: pd.Series,
    minimum_stock_count: int = 30,
) -> float:
    """Calculate one cross-sectional Spearman Rank IC value.
    计算单期横截面的 Spearman Rank IC。

    Rank IC is the correlation between cross-sectional factor ranks and future
    return ranks at a single rebalance date.
    Rank IC 衡量单个调仓日横截面因子排名与未来收益排名之间的相关性。

    Args:
        factor_series (pd.Series): Cross-sectional factor values at one date.
            某一时点的横截面因子值。
        future_return_series (pd.Series): Cross-sectional forward returns for
            the next holding period.
            下一持有期的横截面未来收益。
        minimum_stock_count (int): Minimum valid stock count required to accept
            the IC observation.
            接受该 IC 观测所需的最小有效股票数量。

    Returns:
        float: Spearman Rank IC, or ``np.nan`` when the sample is too small.
        Spearman Rank IC；若样本数量过少，则返回 ``np.nan``。
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
    """Calculate monthly Rank IC values for all rebalance dates.
    为所有调仓日计算月度 Rank IC。

    This replaces the explicit month loop with row-wise DataFrame ranking and
    corrwith(axis=1).
    这里用逐行排名和 ``corrwith(axis=1)`` 替代显式月度循环。

    Args:
        factor_monthly_df (pd.DataFrame): Monthly factor matrix indexed by
            rebalance date with ticker columns.
            调仓日索引、ticker 列的月度因子矩阵。
        future_return_df (pd.DataFrame): Forward return matrix aligned to the
            factor dates and ticker columns.
            与因子日期和 ticker 对齐的未来收益矩阵。
        minimum_stock_count (int): Minimum valid names required for one monthly
            IC observation.
            单个月度 IC 观测所需的最小有效股票数量。

    Returns:
        pd.Series: Monthly Rank IC series named
        ``"RankIC_ResidualVol_Monthly"``.
        名为 ``"RankIC_ResidualVol_Monthly"`` 的月度 Rank IC 序列。
    """
    if factor_monthly_df.empty or future_return_df.empty:
        return pd.Series(dtype="float64", name="RankIC_ResidualVol_Monthly")

    factor_df, future_df = factor_monthly_df.align(future_return_df, join="inner", axis=0)
    factor_df, future_df = factor_df.align(future_df, join="inner", axis=1)
    if factor_df.empty:
        return pd.Series(dtype="float64", name="RankIC_ResidualVol_Monthly")

    valid = factor_df.notna() & future_df.notna()
    valid_count = valid.sum(axis=1)

    # Rank IC is a cross-sectional statistic. With too few names, one outlier
    # can flip the entire rank correlation, so small monthly samples are treated
    # as missing rather than noisy evidence.
    # Rank IC 是横截面统计量。股票数太少时，一个异常点就可能反转整体相关性，
    # 因此小样本月份被视为缺失值，而不是噪声很大的证据。
    factor_rank = factor_df.where(valid).rank(axis=1, method="average")
    future_rank = future_df.where(valid).rank(axis=1, method="average")

    ic_series = factor_rank.corrwith(future_rank, axis=1)
    ic_series = ic_series.where(valid_count >= minimum_stock_count)
    ic_series.name = "RankIC_ResidualVol_Monthly"
    return ic_series.dropna()


def calculate_ic_statistics(
    ic_series: pd.Series,
    frequency: int = 12,
) -> dict[str, float | int]:
    """Calculate common IC statistics.
    计算常见 IC 统计指标。

    Metrics include IC mean/std, monthly ICIR, annualized ICIR, t-stat and
    positive-IC hit rate.
    指标包括 IC 均值/标准差、月度 ICIR、年化 ICIR、t 统计量和正 IC 胜率。

    Args:
        ic_series (pd.Series): Rank IC observations. Rank IC 观测序列。
        frequency (int): Number of IC observations per year for annualization.
            年化使用的每年 IC 观测频率。

    Returns:
        dict[str, float | int]: Dictionary containing sample count, IC mean,
        IC standard deviation, monthly ICIR, annualized ICIR, t-statistic, and
        positive-IC hit rate.
        包含样本数、IC 均值、IC 标准差、月度 ICIR、年化 ICIR、t 统计量和
        正 IC 胜率的字典。
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
    t_value = (
        ic_mean / (ic_std / np.sqrt(count))
        if pd.notna(ic_std) and ic_std != 0 and count > 1
        else np.nan
    )
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


# Group construction / 分组构建


def validate_direction(direction: str) -> None:
    """Validate factor direction text.
    校验因子方向文本。

    Args:
        direction (str): Factor direction. Valid values are
            ``"low_is_good"`` and ``"high_is_good"``.
            因子方向；合法值为 ``"low_is_good"`` 和 ``"high_is_good"``。

    Returns:
        None. 无返回值。

    Raises:
        ValueError: Raised when ``direction`` is unsupported.
        当 ``direction`` 不受支持时抛出。
    """
    if direction not in {"low_is_good", "high_is_good"}:
        raise ValueError('direction must be either "low_is_good" or "high_is_good"')


def build_group_labels(
    factor_series: pd.Series,
    group_count: int = 5,
    direction: str = "low_is_good",
) -> pd.Series:
    """Split stocks into quantile groups by factor value.
    按因子值将股票切分为分位数组。

    Group1 always contains the lowest factor values and GroupN the highest
    factor values. The direction parameter is kept explicit for interpretation
    and for downstream recommendation sorting.
    Group1 始终代表最低因子值组，GroupN 始终代表最高因子值组。``direction``
    参数保留给解释和推荐表内部排序使用。

    Args:
        factor_series (pd.Series): Cross-sectional factor values. 横截面因子值。
        group_count (int): Number of quantile groups. 分位数组数量。
        direction (str): Factor direction used for interpretation. 因子方向解释。

    Returns:
        pd.Series: Integer group labels indexed by ticker. Group1 is the lowest
        factor bucket and GroupN is the highest factor bucket.
        ticker 索引的整数分组标签；Group1 为最低因子组，GroupN 为最高因子组。
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
    """Build one rebalance-date equal-weight grouped portfolio table.
    构建单个调仓日的等权分组组合表。

    Args:
        factor_series (pd.Series): Cross-sectional factor values at one
            rebalance date.
            某个调仓日的横截面因子值。
        rebalance_date_str (str): Rebalance date formatted as ``YYYYMMDD``.
            ``YYYYMMDD`` 格式的调仓日期。
        group_count (int): Number of quantile groups. 分位数组数量。
        minimum_stock_count (int): Minimum valid factor count required before
            portfolios are formed.
            构建组合前所需的最小有效因子数量。
        direction (str): Factor direction used for interpretation. 因子方向解释。

    Returns:
        pd.DataFrame: Long-format portfolio table with columns ``日期``,
        ``代码``, ``权重``, and ``组合名称``.
        长表形式的组合表，包含 ``日期``、``代码``、``权重`` 和 ``组合名称``。
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
    """Build long-format grouped portfolio weights for all rebalance dates.
    为所有调仓日构建长表形式的分组组合权重。

    Args:
        factor_monthly_df (pd.DataFrame): Monthly factor matrix indexed by
            rebalance date with ticker columns.
            调仓日索引、ticker 列的月度因子矩阵。
        group_count (int): Number of quantile groups. 分位数组数量。
        minimum_stock_count (int): Minimum valid factor count required at each
            rebalance date.
            每个调仓日所需的最小有效因子数量。
        direction (str): Factor direction used for interpretation. 因子方向解释。

    Returns:
        pd.DataFrame: Long-format grouped portfolio weights for all valid
        rebalance dates.
        所有有效调仓日的长表形式分组组合权重。
    """
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


# Monthly group backtest / 月度分组回测


def empty_monthly_group_backtest_result() -> MonthlyGroupBacktestResult:
    """Create an empty monthly group backtest result.
    构建空的月度分组回测结果。

    Args:
        None. 无参数。

    Returns:
        MonthlyGroupBacktestResult: Result object whose DataFrames are empty.
        所有 DataFrame 均为空的回测结果对象。
    """
    empty_df = pd.DataFrame()
    return MonthlyGroupBacktestResult(
        nav_df=empty_df,
        gross_nav_df=empty_df,
        period_return_df=empty_df,
        gross_return_df=empty_df,
        turnover_df=empty_df,
        transaction_cost_df=empty_df,
        schedule_df=empty_df,
    )


def calculate_monthly_group_backtest_details(
    monthly_close_df: pd.DataFrame,
    monthly_portfolio_df: pd.DataFrame,
    signal_lag_periods: int = 1,
    transaction_cost_bps: float = 10.0,
) -> MonthlyGroupBacktestResult:
    """Backtest monthly grouped portfolios with lag, turnover, and costs.
    使用信号滞后、换手率和交易成本回测月度分组组合。

    ``signal_lag_periods=1`` means a signal observed at month-end t is first
    applied to the return from t+1 to t+2. With month-end-only data, this is a
    conservative way to avoid assuming that we can observe month-end close data
    and trade at that same close.
    ``signal_lag_periods=1`` 表示 t 月末观察到的信号，最早用于 t+1 到 t+2
    的收益。由于这里只使用月末价格，这是一种偏保守的处理方式，避免假设我们能
    在看到月末收盘价后又以同一个收盘价完成交易。

    Transaction cost is charged on absolute turnover. Initial entry from cash
    has turnover close to 1.0 for each fully invested group.
    交易成本按绝对换手率扣减；从现金首次建仓时，满仓组合的换手率约为 1.0。

    Args:
        monthly_close_df (pd.DataFrame): Month-end close-price matrix indexed by
            date with ticker columns.
            日期索引、ticker 列的月末收盘价矩阵。
        monthly_portfolio_df (pd.DataFrame): Long-format portfolio table with
            columns ``日期``, ``代码``, ``权重``, and ``组合名称``.
            长表形式组合表，包含 ``日期``、``代码``、``权重`` 和 ``组合名称``。
        signal_lag_periods (int): Number of monthly periods between signal date
            and holding-period start date.
            信号日期和持有期开始日期之间的月度滞后期数。
        transaction_cost_bps (float): One-way transaction cost in basis points
            applied to absolute turnover.
            按绝对换手率扣减的单边交易成本，单位为基点。

    Returns:
        MonthlyGroupBacktestResult: Net NAV, gross NAV, returns, turnover,
        transaction cost, and signal schedule.
        包含净值、毛净值、收益率、换手率、交易成本和信号排期的结果对象。

    Raises:
        ValueError: Raised when lag or transaction cost is negative.
        当滞后期数或交易成本为负时抛出。
    """
    if signal_lag_periods < 0:
        raise ValueError("signal_lag_periods must be >= 0")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if monthly_close_df.empty or monthly_portfolio_df.empty:
        return empty_monthly_group_backtest_result()

    close_df = downcast_float_frame(monthly_close_df).sort_index()
    portfolio_df = monthly_portfolio_df.copy()
    portfolio_df["rebalance_dt"] = pd.to_datetime(
        portfolio_df["日期"].astype(str),
        errors="coerce",
    )
    portfolio_df["代码"] = portfolio_df["代码"].astype(str)
    portfolio_df["组合名称"] = portfolio_df["组合名称"].astype(str)
    portfolio_df = portfolio_df.dropna(subset=["rebalance_dt"])
    portfolio_df = portfolio_df[portfolio_df["rebalance_dt"].isin(close_df.index)]

    if portfolio_df.empty:
        return empty_monthly_group_backtest_result()

    close_dates = pd.DatetimeIndex(close_df.index).sort_values()
    available_signal_dates = set(pd.DatetimeIndex(portfolio_df["rebalance_dt"].unique()))
    schedule_records: list[dict[str, pd.Timestamp]] = []

    for period_start_pos in range(signal_lag_periods, len(close_dates) - 1):
        signal_dt = close_dates[period_start_pos - signal_lag_periods]
        period_start_dt = close_dates[period_start_pos]
        period_end_dt = close_dates[period_start_pos + 1]
        if signal_dt in available_signal_dates:
            schedule_records.append(
                {
                    "signal_dt": signal_dt,
                    "period_start_dt": period_start_dt,
                    "period_end_dt": period_end_dt,
                }
            )

    if not schedule_records:
        return empty_monthly_group_backtest_result()

    schedule_df = pd.DataFrame(schedule_records)
    period_start_index = pd.DatetimeIndex(schedule_df["period_start_dt"])
    period_end_index = pd.DatetimeIndex(schedule_df["period_end_dt"])
    signal_index = pd.DatetimeIndex(schedule_df["signal_dt"])
    group_names = sorted(portfolio_df["组合名称"].unique().tolist())

    gross_return_df = pd.DataFrame(index=period_start_index, columns=group_names)
    period_return_df = pd.DataFrame(index=period_start_index, columns=group_names)
    turnover_df = pd.DataFrame(index=period_start_index, columns=group_names)
    transaction_cost_df = pd.DataFrame(index=period_start_index, columns=group_names)
    cost_rate_per_turnover = transaction_cost_bps / 10000.0

    for group_name in group_names:
        group_holdings = portfolio_df[portfolio_df["组合名称"] == group_name]
        if group_holdings.empty:
            continue

        weights_by_signal = group_holdings.pivot_table(
            index="rebalance_dt",
            columns="代码",
            values="权重",
            aggfunc="sum",
            observed=True,
        )
        target_weights = weights_by_signal.reindex(signal_index).fillna(0.0)
        target_weights.index = period_start_index
        target_weights = downcast_float_frame(target_weights)

        start_prices = close_df.reindex(
            index=period_start_index,
            columns=target_weights.columns,
        )
        end_prices = close_df.reindex(
            index=period_end_index,
            columns=target_weights.columns,
        )
        end_prices.index = period_start_index
        asset_return_df = (end_prices / start_prices - 1.0).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        asset_return_df = asset_return_df.fillna(0.0)
        gross_return_series = (target_weights * asset_return_df).sum(axis=1)

        turnover_values: list[float] = []
        previous_target: pd.Series | None = None
        previous_start_dt: pd.Timestamp | None = None

        for period_start_dt, target in target_weights.iterrows():
            target = pd.to_numeric(target, errors="coerce").fillna(0.0)
            if previous_target is None or previous_start_dt is None:
                drifted_weight = target * 0.0
            else:
                drift_return = (
                    close_df.loc[period_start_dt, target.index]
                    / close_df.loc[previous_start_dt, target.index]
                    - 1.0
                )
                drift_return = pd.to_numeric(drift_return, errors="coerce")
                drift_return = drift_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                previous_target = previous_target.reindex(target.index).fillna(0.0)
                gross_to_rebalance = float((previous_target * drift_return).sum())
                denominator = 1.0 + gross_to_rebalance
                if pd.isna(denominator) or denominator <= 0:
                    drifted_weight = previous_target * 0.0
                else:
                    drifted_weight = previous_target * (1.0 + drift_return) / denominator

            turnover_values.append(float((target - drifted_weight).abs().sum()))
            previous_target = target
            previous_start_dt = period_start_dt

        turnover_series = pd.Series(turnover_values, index=period_start_index)
        transaction_cost_series = turnover_series * cost_rate_per_turnover
        net_return_series = (
            (1.0 - transaction_cost_series) * (1.0 + gross_return_series) - 1.0
        )

        gross_return_df[group_name] = gross_return_series.astype("float32")
        period_return_df[group_name] = net_return_series.astype("float32")
        turnover_df[group_name] = turnover_series.astype("float32")
        transaction_cost_df[group_name] = transaction_cost_series.astype("float32")

    gross_nav_after = (1.0 + gross_return_df.fillna(0.0)).cumprod()
    nav_after = (1.0 + period_return_df.fillna(0.0)).cumprod()
    gross_nav_after.index = period_end_index
    nav_after.index = period_end_index

    initial_index = [period_start_index[0]]
    gross_initial_nav = pd.DataFrame(
        {group_name: 1.0 for group_name in group_names},
        index=initial_index,
    )
    initial_nav = gross_initial_nav.copy()

    gross_nav_df = pd.concat([gross_initial_nav, gross_nav_after], axis=0)
    nav_df = pd.concat([initial_nav, nav_after], axis=0)
    gross_nav_df.index.name = "date"
    nav_df.index.name = "date"

    return MonthlyGroupBacktestResult(
        nav_df=downcast_float_frame(nav_df),
        gross_nav_df=downcast_float_frame(gross_nav_df),
        period_return_df=downcast_float_frame(period_return_df),
        gross_return_df=downcast_float_frame(gross_return_df),
        turnover_df=downcast_float_frame(turnover_df),
        transaction_cost_df=downcast_float_frame(transaction_cost_df),
        schedule_df=schedule_df,
    )


def backtest_monthly_groups(
    monthly_close_df: pd.DataFrame,
    monthly_portfolio_df: pd.DataFrame,
    signal_lag_periods: int = 1,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest monthly groups and return net NAV for backward compatibility.
    回测月度分组组合，并返回净值矩阵以兼容旧调用方式。

    Args:
        monthly_close_df (pd.DataFrame): Month-end close-price matrix indexed by
            date with ticker columns.
            日期索引、ticker 列的月末收盘价矩阵。
        monthly_portfolio_df (pd.DataFrame): Long-format portfolio table with
            columns ``日期``, ``代码``, ``权重``, and ``组合名称``.
            长表形式组合表，包含 ``日期``、``代码``、``权重`` 和 ``组合名称``。
        signal_lag_periods (int): Number of monthly periods between signal date
            and holding-period start date.
            信号日期和持有期开始日期之间的月度滞后期数。
        transaction_cost_bps (float): One-way transaction cost in basis points.
            单边交易成本，单位为基点。

    Returns:
        pd.DataFrame: Net group NAV matrix indexed by date.
        日期索引的分组净值矩阵。
    """
    result = calculate_monthly_group_backtest_details(
        monthly_close_df=monthly_close_df,
        monthly_portfolio_df=monthly_portfolio_df,
        signal_lag_periods=signal_lag_periods,
        transaction_cost_bps=transaction_cost_bps,
    )
    return result.nav_df


# Winner diagnostics and recommendation output / 赢家组诊断与推荐输出


def diagnose_recent_winner_groups(
    nav_df: pd.DataFrame,
    recent_months: int = 12,
) -> pd.Series:
    """Count which group has the highest NAV in recent observations.
    统计最近观测期内哪个分组净值最高。

    Note: this measures the current leading NAV group, not the highest one-month
    return group. It is intended as a simple heuristic for the recommendation step.
    注意：这里衡量的是当前累计净值领先的组，而不是单月收益最高的组。它只是给
    推荐步骤提供一个简单启发式规则。

    Args:
        nav_df (pd.DataFrame): Group NAV matrix indexed by date. 日期索引的分组净值矩阵。
        recent_months (int): Number of latest observations used for the count.
            用于统计的最近观测期数。

    Returns:
        pd.Series: Winner-group counts sorted descending. 赢家组出现次数，降序排列。
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
    """Build the latest recommendation table from the backtest winner group.
    基于回测赢家组构建最新一期推荐表。

    Inside the selected group:
    - low_is_good: pick the lowest factor values
    - high_is_good: pick the highest factor values
    在选定分组内部：
    - low_is_good：选择因子值最低的股票
    - high_is_good：选择因子值最高的股票

    Args:
        as_of_month_end (pd.Timestamp): Latest factor date used for selection.
            用于选股的最新因子日期。
        winner_group (str): Group name selected by recent backtest diagnostics.
            由近期回测诊断选出的分组名称。
        factor_monthly_df (pd.DataFrame): Monthly factor matrix. 月度因子矩阵。
        monthly_close_df (pd.DataFrame): Month-end close-price matrix. 月末收盘价矩阵。
        daily_close_df (pd.DataFrame): Daily close-price matrix after filtering.
            过滤后的日频收盘价矩阵。
        direction (str): Factor direction used for within-group sorting.
            分组内部排序使用的因子方向。
        top_n (int): Maximum number of tickers to recommend. 推荐股票数量上限。

    Returns:
        pd.DataFrame: Recommendation table with factor value, equal weight,
        trailing returns, and trailing annualized volatility.
        推荐表，包含因子值、等权权重、历史收益和近一年年化波动率。
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

    # Momentum and trailing risk are not part of the factor sort; they are shown
    # to make the final recommendation auditable before any manual decision.
    # 动量和历史风险不参与因子排序；展示它们是为了让人工决策前能快速审阅候选股。
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
            "rule": [f"Winner group = {winner_group} -> pick top_n within group"]
            * len(selected_tickers),
        }
    )

    for col in ["winner_group_from_backtest", "ticker", "group"]:
        report_df[col] = report_df[col].astype("category")
    return report_df


# Plotting helpers / 绘图工具


def plot_monthly_ic_report(
    ic_series: pd.Series,
    title_prefix: str = "Monthly Rank IC",
) -> None:
    """Plot IC bar chart, IC line chart, and cumulative IC chart.
    绘制 IC 柱状图、IC 折线图和累计 IC 图。

    Args:
        ic_series (pd.Series): Monthly Rank IC series. 月度 Rank IC 序列。
        title_prefix (str): Prefix used in chart titles. 图表标题前缀。

    Returns:
        None. 无返回值。
    """
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
    """Plot grouped portfolio NAV curves.
    绘制分组组合净值曲线。

    Args:
        nav_df (pd.DataFrame): Group NAV matrix indexed by date. 日期索引的分组净值矩阵。
        title (str): Chart title. 图表标题。

    Returns:
        None. 无返回值。
    """
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


def plot_long_short_nav(
    nav_df: pd.DataFrame,
    long_group: str = "Group5",
    short_group: str = "Group1",
) -> None:
    """Plot long-short NAV as ``long_group / short_group``.
    绘制 ``long_group / short_group`` 形式的多空净值。

    Args:
        nav_df (pd.DataFrame): Group NAV matrix indexed by date. 日期索引的分组净值矩阵。
        long_group (str): Group used as the long leg. 多头组名称。
        short_group (str): Group used as the short leg. 空头组名称。

    Returns:
        None. 无返回值。
    """
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


# Workflow orchestration / 流程编排


def run_residual_volatility_backtest(
    ticker_list: Sequence[str],
    data_config: DataDownloadConfig | None = None,
    backtest_config: ResidualVolatilityBacktestConfig | None = None,
    plot: bool = True,
    save_outputs: bool = False,
    output_dir: str | Path = "outputs",
) -> BacktestResult | None:
    """Run the full residual-volatility single-factor research workflow.
    运行完整的残差波动率单因子研究流程。

    Steps:
    1. Download/cache stock and benchmark close prices
    2. Filter stock pool
    3. Compute monthly residual volatility factor
    4. Calculate Rank IC
    5. Build grouped monthly portfolios, turnover, transaction cost, and NAV
    6. Produce latest recommendation table
    步骤：
    1. 下载/缓存股票和基准收盘价
    2. 过滤股票池
    3. 计算月度残差波动率因子
    4. 计算 Rank IC
    5. 构建月度分组组合，并计算换手率、交易成本和净值
    6. 生成最新一期推荐表

    Args:
        ticker_list (Sequence[str]): Stock universe candidates. 候选股票池 ticker 列表。
        data_config (DataDownloadConfig | None): Optional data download and
            cache settings.
            可选的数据下载和缓存配置。
        backtest_config (ResidualVolatilityBacktestConfig | None): Optional
            factor, filter, IC, and backtest settings.
            可选的因子、过滤、IC 和回测配置。
        plot (bool): Whether to display IC and NAV charts. 是否展示 IC 和净值图。
        save_outputs (bool): Whether to persist CSV outputs. 是否保存 CSV 输出。
        output_dir (str | Path): Directory for optional CSV outputs. CSV 输出目录。

    Returns:
        BacktestResult | None: Structured workflow outputs, or ``None`` when
        required data/intermediate results are unavailable.
        结构化流程结果；若关键数据或中间结果不可用，则返回 ``None``。
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

    print(
        f"价格矩阵: {close_price_df.shape}, "
        f"memory={memory_usage_mb(close_price_df):.2f} MB"
    )

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
        print(
            removed_df.head(25)[
                ["ticker", "reason", "trading_days", "last_price", "ann_vol_12m"]
            ].to_string(index=False)
        )

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
        plot_monthly_ic_report(
            ic_series,
            title_prefix="Monthly Rank IC (Residual Volatility Factor)",
        )

    monthly_portfolio_df = build_monthly_equal_weight_group_portfolios(
        factor_monthly_df=factor_monthly_df,
        group_count=backtest_config.group_count,
        minimum_stock_count=backtest_config.minimum_group_stock_count,
        direction=backtest_config.direction,
    )

    if monthly_portfolio_df.empty:
        print("portfolio 为空：可用股票太少或因子缺失太多。")
        return None

    group_backtest_result = calculate_monthly_group_backtest_details(
        monthly_close_df=monthly_close_df,
        monthly_portfolio_df=monthly_portfolio_df,
        signal_lag_periods=backtest_config.signal_lag_periods,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
    )
    nav_df = group_backtest_result.nav_df

    print_divider("净值（最后5行）")
    if nav_df.empty:
        print("Empty NAV dataframe.")
        return None
    print(nav_df.tail())

    print_divider("换手率与交易成本（最后5行）")
    print(
        f"信号滞后期数: {backtest_config.signal_lag_periods} | "
        f"单边交易成本: {backtest_config.transaction_cost_bps:.2f} bps"
    )
    if group_backtest_result.turnover_df.empty:
        print("Empty turnover dataframe.")
    else:
        print("--- Turnover ---")
        print(group_backtest_result.turnover_df.tail())
    if group_backtest_result.transaction_cost_df.empty:
        print("Empty transaction cost dataframe.")
    else:
        print("--- Transaction Cost Drag ---")
        print(group_backtest_result.transaction_cost_df.tail())

    if plot:
        plot_group_nav(
            nav_df,
            title="Group NAV (Residual Volatility Factor, Monthly Rebalance)",
        )
        plot_long_short_nav(nav_df)

    winner_count = diagnose_recent_winner_groups(
        nav_df,
        recent_months=backtest_config.recent_winner_months,
    )
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
        print(
            f"调仓月末: {latest_month_end.date()} | "
            f"赢家组: {winner_group} | "
            f"等权持仓数: {len(recommendation_df)}"
        )
        print("Tickers:", recommendation_df["ticker"].astype(str).tolist())

        print_divider("推荐表（基于：回测赢家组 + 最新一期因子分组）")
        display_df = recommendation_df.copy()
        display_df["weight_eq"] = display_df["weight_eq"].apply(lambda x: format_percent(x, 2))
        display_df["ret_3m"] = display_df["ret_3m"].apply(lambda x: format_percent(x, 2))
        display_df["ret_12m"] = display_df["ret_12m"].apply(lambda x: format_percent(x, 2))
        display_df["ann_vol_12m"] = display_df["ann_vol_12m"].apply(
            lambda x: format_percent(x, 2)
        )
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
        gross_nav_df=group_backtest_result.gross_nav_df,
        turnover_df=group_backtest_result.turnover_df,
        transaction_cost_df=group_backtest_result.transaction_cost_df,
        backtest_schedule_df=group_backtest_result.schedule_df,
        recommendation_df=recommendation_df,
    )

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ic_series.to_csv(output_path / "ic_series.csv", encoding="utf-8-sig")
        nav_df.to_csv(output_path / "group_nav.csv", encoding="utf-8-sig")
        group_backtest_result.gross_nav_df.to_csv(
            output_path / "group_gross_nav.csv",
            encoding="utf-8-sig",
        )
        group_backtest_result.turnover_df.to_csv(
            output_path / "group_turnover.csv",
            encoding="utf-8-sig",
        )
        group_backtest_result.transaction_cost_df.to_csv(
            output_path / "group_transaction_cost.csv",
            encoding="utf-8-sig",
        )
        group_backtest_result.schedule_df.to_csv(
            output_path / "group_backtest_schedule.csv",
            index=False,
            encoding="utf-8-sig",
        )
        recommendation_df.to_csv(
            output_path / "recommendation_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
        filter_summary_df.to_csv(
            output_path / "filter_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"已保存结果到: {output_path.resolve()}")

    return result


# Script entry point / 脚本入口


def main() -> None:
    """Run the residual-volatility backtest with the default ticker pool.
    使用默认 ticker 池运行残差波动率回测。

    Args:
        None. 无参数。

    Returns:
        None. 无返回值。
    """
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
        signal_lag_periods=1,
        transaction_cost_bps=10.0,
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
