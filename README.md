# US Stock Quant: Residual Volatility Factor Strategy

🌍 *[简体中文](README_zh-CN.md) ∙ [English](README.md)*

This project is an end-to-end US stock quantitative backtesting and stock-picking script based on the **Residual Volatility** factor. It utilizes [AkShare](https://akshare.xyz/) to fetch daily US stock market data, implementing a fully automated pipeline from data acquisition and factor construction to portfolio backtesting and final stock recommendations.

## Core Features

- **Automated Data Fetching**: Retrieves daily closing prices for a specified US stock pool and the market benchmark (SPY) via AkShare, featuring automatic error handling and data alignment.
- **Smart Stock Pool Filtering**: Filters out unqualified stocks based on historical trading days (>=504), absolute stock price ($5 - $150), and annualized volatility (<=80%).
- **Factor Construction**: Calculates the monthly Residual Volatility factor based on the CAPM model, running a linear regression of daily stock returns against the SPY benchmark over a 63-day rolling window.
- **Factor Evaluation**: Computes the monthly Spearman Rank IC and outputs comprehensive statistics including IC Mean, Standard Deviation, Annualized ICIR, t-statistic, and Win Rate.
- **Monthly Backtesting**: Divides the stock pool into 5 quantiles based on factor exposure. Simulates a monthly-rebalanced, equal-weighted portfolio, generating historical Net Asset Value (NAV) and Long-Short spread curves.
- **Intelligent Stock Recommendation**: Diagnoses the "Winner Group" that performed best over the past 12 months, and selects the Top 10 stocks within this group based on the latest factor values to provide actionable stock picks.

## Prerequisites

Ensure you have Python 3.9+ installed along with the following dependencies:

```bash
pip install numpy pandas akshare matplotlib
```

## Usage

Run the main script to start the backtesting and stock-picking process:

```bash
python main.py
```
