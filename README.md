# US Stock Quant: Residual Volatility Factor Strategy
*(Scroll down for the Chinese version / 中文版请向下滚动)*

This project is an end-to-end US stock quantitative backtesting and stock-picking script based on the **Residual Volatility** factor. It utilizes [AkShare](https://akshare.xyz/) to fetch daily US stock market data, implementing a fully automated pipeline from data acquisition and factor construction to portfolio backtesting and final stock recommendations.

## 🚀 Core Features

* **Automated Data Fetching**: Retrieves daily closing prices for a specified US stock pool and the market benchmark (SPY) via AkShare, featuring automatic error handling and data alignment.
* **Smart Stock Pool Filtering**: Filters out unqualified stocks based on historical trading days (>=504), absolute stock price ($5 - $150), and annualized volatility (<=80%).
* **Factor Construction**: Calculates the monthly Residual Volatility factor based on the CAPM model, running a linear regression of daily stock returns against the SPY benchmark over a 63-day rolling window.
* **Factor Evaluation**: Computes the monthly Spearman Rank IC and outputs comprehensive statistics including IC Mean, Standard Deviation, Annualized ICIR, t-statistic, and Win Rate.
* **Monthly Backtesting**: Divides the stock pool into 5 quantiles based on factor exposure. Simulates a monthly-rebalanced, equal-weighted portfolio, generating historical Net Asset Value (NAV) and Long-Short spread curves.
* **Intelligent Stock Recommendation**: Diagnoses the "Winner Group" that performed best over the past 12 months, and selects the Top 10 stocks within this group based on the latest factor values to provide actionable stock picks.

## 🛠️ Prerequisites

Ensure you have Python 3.9+ installed along with the following dependencies:

```bash
pip install numpy pandas akshare matplotlib

# US Stock Quant: Residual Volatility Factor Strategy

本项目是一个完整的美股单因子量化回测与选股流程脚本。以**残差波动率（Residual Volatility）**为核心因子，利用 [AkShare](https://akshare.xyz/) 获取免费的美股日频行情数据，实现了从数据获取、因子计算、有效性检验到分组回测及最终实盘选股的端到端（End-to-End）全自动化流程。

## 🚀 核心功能 (Core Features)

* **自动化数据源集成**：使用 AkShare 自动拉取指定美股股票池及基准（SPY）的日线收盘价，并具备多版本接口容错与自动对齐机制。
* **智能股票池清洗**：按历史交易天数（>=504天）、绝对股价（$5 - $150）以及年化波动率（<=80%）自动剔除不合规标的。
* **因子计算 (Factor Construction)**：基于 CAPM 模型，使用过去 63 个交易日的日收益率与 SPY 市场基准进行线性回归，提取月度残差波动率因子。
* **因子有效性检验 (Factor Evaluation)**：计算每月的 Spearman Rank IC，并输出完整的 IC 统计指标（IC均值、标准差、年化 ICIR、t统计量及胜率）。
* **月频分组回测 (Monthly Backtesting)**：按因子暴露度将股票等分为 5 组（等权持仓，每月月末调仓），计算各组历史净值，并生成直观的多空净值对比曲线。
* **智能选股推荐 (Stock Recommendation)**：诊断过去 12 个月内表现最佳的“赢家组（Winner Group）”，并在该组内按因子最新值精选 Top 10 股票，生成实盘调仓建议表。

## 🛠️ 环境依赖 (Prerequisites)

请确保你的本地环境中安装了 Python 3.9+，并安装以下依赖包：

```bash
pip install numpy pandas akshare matplotlib
