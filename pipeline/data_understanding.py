from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
MODEL_INPUT_SUMMARY_FILE = DATA_DIR / "model-input-summary.csv"

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_UNDERSTANDING_SUMMARY_DIR = SUMMARY_DIR / "data-understanding"
EDA_CHARTS_DIR = DATA_UNDERSTANDING_SUMMARY_DIR / "eda-charts"

COLUMN_MEANINGS = {
    "Date": "Trading date for the daily stock record.",
    "Ticker": "Stock ticker symbol from Yahoo Finance.",
    "Open": "Opening price for the trading day.",
    "High": "Highest traded price for the trading day.",
    "Low": "Lowest traded price for the trading day.",
    "Close": "Closing price for the trading day.",
    "Adj Close": "Adjusted closing price after dividends and splits.",
    "Volume": "Number of shares traded for the day.",
    "Dividends": "Dividend amount recorded for the trading day.",
    "Stock Splits": "Stock split ratio recorded for the trading day.",
}

NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


# ============================================================
# Helpers
# ============================================================
def ensure_summary_structure() -> None:
    DATA_UNDERSTANDING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessed_csv_files() -> List[Path]:
    return sorted(PREPROCESSED_DATA_DIR.glob("*.csv"))


def get_model_input_csv_files() -> List[Path]:
    return sorted(MODEL_INPUT_DATA_DIR.glob("*.csv"))


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_No rows to report._"

    string_rows = [[format_number(value) for value in row] for row in rows]
    column_widths = [
        max(3, len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: List[str]) -> str:
        padded_values = [
            value.ljust(column_widths[index])
            for index, value in enumerate(values)
        ]
        return "| " + " | ".join(padded_values) + " |"

    lines = [
        format_row(headers),
        "| " + " | ".join("-" * width for width in column_widths) + " |",
    ]
    for row in string_rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def get_matplotlib_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate EDA chart PNG files. "
            "Install it in the project environment with: "
            ".\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    return plt, FuncFormatter


def load_stock_files(files: List[Path]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["Source_File"] = file_path.name
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    return combined_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# 3.4.2.1 Variables
# ============================================================
def write_variables_summary(df: pd.DataFrame) -> None:
    rows: List[List[Any]] = []
    for col in df.columns:
        if col == "Source_File":
            continue

        series = df[col]
        min_value: Any = "N/A"
        max_value: Any = "N/A"
        if pd.api.types.is_numeric_dtype(series):
            min_value = series.min()
            max_value = series.max()
        elif pd.api.types.is_datetime64_any_dtype(series):
            min_value = series.min().date() if not series.dropna().empty else "N/A"
            max_value = series.max().date() if not series.dropna().empty else "N/A"

        rows.append([
            col,
            str(series.dtype),
            COLUMN_MEANINGS.get(col, "Generated or supporting dataset field."),
            int(series.isna().sum()),
            int(series.nunique(dropna=True)),
            min_value,
            max_value,
        ])

    content = f"""# 3.4.2.1 Variables Summary

## Purpose
This section documents the variables available in the preprocessed stock dataset and describes their basic data characteristics.

## Variable Dictionary
{markdown_table(
        [
            "Variable",
            "Data Type",
            "Meaning",
            "Missing Count",
            "Unique Count",
            "Minimum",
            "Maximum",
        ],
        rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "variables-summary.md", content)


# ============================================================
# 3.4.2.2 Observations
# ============================================================
def write_observations_summary(df: pd.DataFrame) -> None:
    rows: List[List[Any]] = []
    for ticker, ticker_df in df.groupby("Ticker", dropna=False):
        valid_dates = ticker_df["Date"].dropna()
        duplicate_dates = int(ticker_df.duplicated(subset=["Ticker", "Date"]).sum())

        if valid_dates.empty:
            first_date: Any = "N/A"
            last_date: Any = "N/A"
            business_days = 0
            coverage_rate: Any = "N/A"
        else:
            first_date = valid_dates.min().date()
            last_date = valid_dates.max().date()
            business_days = len(pd.bdate_range(valid_dates.min(), valid_dates.max()))
            coverage_rate = (
                len(valid_dates) / business_days
                if business_days > 0
                else "N/A"
            )

        rows.append([
            ticker,
            int(len(ticker_df)),
            first_date,
            last_date,
            duplicate_dates,
            business_days,
            coverage_rate,
        ])

    content = f"""# 3.4.2.2 Observations Summary

## Purpose
This section documents the observation coverage of the preprocessed dataset by stock ticker.

## Notes
Trading-day coverage is compared against weekday business days only. Public holidays and market closures are not separately removed from that denominator, so this value is an approximate coverage indicator.

## Observation Coverage
{markdown_table(
        [
            "Ticker",
            "Rows",
            "First Date",
            "Last Date",
            "Duplicate Dates",
            "Weekday Business Days",
            "Approx Coverage Rate",
        ],
        rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "observations-summary.md", content)


# ============================================================
# 3.4.2.3 Exploratory Data Analysis
# ============================================================
def build_descriptive_rows(df: pd.DataFrame) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append([
            col,
            int(series.count()),
            series.mean(),
            series.std(),
            series.min(),
            series.quantile(0.25),
            series.median(),
            series.quantile(0.75),
            series.max(),
        ])
    return rows


def build_return_rows(df: pd.DataFrame) -> List[List[Any]]:
    working_df = df.sort_values(["Ticker", "Date"]).copy()
    working_df["Daily_Return"] = working_df.groupby("Ticker")["Adj Close"].pct_change()

    rows: List[List[Any]] = []
    for ticker, ticker_df in working_df.groupby("Ticker", dropna=False):
        returns = ticker_df["Daily_Return"].dropna()
        if returns.empty:
            rows.append([ticker, 0, "N/A", "N/A", "N/A", "N/A"])
            continue
        rows.append([
            ticker,
            int(returns.count()),
            returns.mean(),
            returns.std(),
            returns.min(),
            returns.max(),
        ])
    return rows


def build_correlation_rows(df: pd.DataFrame) -> List[List[Any]]:
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    corr_df = df[available_columns].corr(numeric_only=True)
    rows: List[List[Any]] = []

    for row_index, row_name in enumerate(available_columns):
        for col_name in available_columns[row_index + 1:]:
            correlation = corr_df.loc[row_name, col_name]
            if pd.isna(correlation):
                continue
            rows.append([row_name, col_name, float(correlation)])

    rows = sorted(rows, key=lambda item: abs(float(item[2])), reverse=True)
    return rows[:10]


def prepare_eda_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    chart_df = df.sort_values(["Ticker", "Date"]).copy()
    chart_df["Daily_Return"] = chart_df.groupby("Ticker")["Adj Close"].pct_change()
    chart_df["Rolling_Volatility_20D"] = chart_df.groupby("Ticker")[
        "Daily_Return"
    ].transform(lambda series: series.rolling(window=20).std())
    return chart_df


def save_adj_close_trend_chart(df: pd.DataFrame) -> Path:
    plt, _ = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "adj-close-trend.png"

    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker, ticker_df in df.groupby("Ticker"):
        ax.plot(ticker_df["Date"], ticker_df["Adj Close"], linewidth=1.4, label=ticker)

    ax.set_title("Adjusted Closing Price Trend by Ticker")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_monthly_volume_trend_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "monthly-volume-trend.png"

    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker, ticker_df in df.groupby("Ticker"):
        monthly_volume = (
            ticker_df.set_index("Date")["Volume"]
            .resample("ME")
            .mean()
        )
        ax.plot(monthly_volume.index, monthly_volume.values, linewidth=1.4, label=ticker)

    ax.set_title("Monthly Average Trading Volume by Ticker")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Volume")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.0f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_daily_return_distribution_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "daily-return-distribution.png"
    returns = df["Daily_Return"].dropna()

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(returns, bins=80, color="#2f6f9f", edgecolor="white", alpha=0.85)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_rolling_volatility_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "rolling-volatility-20d-trend.png"

    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker, ticker_df in df.groupby("Ticker"):
        ax.plot(
            ticker_df["Date"],
            ticker_df["Rolling_Volatility_20D"],
            linewidth=1.2,
            label=ticker,
        )

    ax.set_title("20-Day Rolling Volatility Trend by Ticker")
    ax.set_xlabel("Date")
    ax.set_ylabel("20-Day Rolling Volatility")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_correlation_heatmap_chart(df: pd.DataFrame) -> Path:
    plt, _ = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "market-feature-correlation-heatmap.png"
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    corr_df = df[available_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap of Market Variables")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)

    for row_index in range(len(corr_df.index)):
        for col_index in range(len(corr_df.columns)):
            value = corr_df.iloc[row_index, col_index]
            if pd.notna(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black",
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_return_vs_volatility_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "return-vs-volatility-by-ticker.png"
    risk_return_df = (
        df.groupby("Ticker")["Daily_Return"]
        .agg(["mean", "std"])
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(risk_return_df["std"], risk_return_df["mean"], s=80, color="#376c5a")
    for _, row in risk_return_df.iterrows():
        ax.annotate(
            row["Ticker"],
            (row["std"], row["mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_title("Average Daily Return vs Daily Return Volatility by Ticker")
    ax.set_xlabel("Daily Return Volatility")
    ax.set_ylabel("Average Daily Return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.2f}%"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_total_return_by_ticker_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "total-return-by-ticker.png"
    rows: List[Dict[str, Any]] = []

    for ticker, ticker_df in df.groupby("Ticker"):
        prices = ticker_df.sort_values("Date")["Adj Close"].dropna()
        if len(prices) < 2:
            continue
        rows.append({
            "Ticker": ticker,
            "Total_Return": (prices.iloc[-1] / prices.iloc[0]) - 1,
        })

    total_return_df = (
        pd.DataFrame(rows)
        .sort_values("Total_Return", ascending=False)
        .reset_index(drop=True)
    )
    colors = [
        "#376c5a" if value >= 0 else "#b45f5f"
        for value in total_return_df["Total_Return"]
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(total_return_df["Ticker"], total_return_df["Total_Return"], color=colors)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("Total Return by Ticker")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Total Return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_annual_return_by_ticker_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "annual-return-by-ticker.png"
    working_df = df.dropna(subset=["Date", "Adj Close"]).copy()
    working_df["Year"] = working_df["Date"].dt.year
    rows: List[Dict[str, Any]] = []

    for (ticker, year), group_df in working_df.groupby(["Ticker", "Year"]):
        prices = group_df.sort_values("Date")["Adj Close"].dropna()
        if len(prices) < 2:
            continue
        rows.append({
            "Ticker": ticker,
            "Year": int(year),
            "Annual_Return": (prices.iloc[-1] / prices.iloc[0]) - 1,
        })

    annual_return_df = pd.DataFrame(rows)
    pivot_df = annual_return_df.pivot(
        index="Ticker",
        columns="Year",
        values="Annual_Return",
    ).sort_index()

    tickers = list(pivot_df.index)
    years = list(pivot_df.columns)
    x_positions = list(range(len(tickers)))
    bar_width = min(0.8 / max(len(years), 1), 0.16)

    fig, ax = plt.subplots(figsize=(13, 7))
    for year_index, year in enumerate(years):
        offsets = [
            position + (year_index - (len(years) - 1) / 2) * bar_width
            for position in x_positions
        ]
        ax.bar(offsets, pivot_df[year], width=bar_width, label=str(year))

    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("Annual Return by Ticker")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Annual Return")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Year", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_daily_return_boxplot_by_ticker_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "daily-return-boxplot-by-ticker.png"
    grouped_returns = [
        ticker_df["Daily_Return"].dropna()
        for _, ticker_df in df.groupby("Ticker")
    ]
    tickers = [
        ticker
        for ticker, ticker_df in df.groupby("Ticker")
        if not ticker_df["Daily_Return"].dropna().empty
    ]
    grouped_returns = [series for series in grouped_returns if not series.empty]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(
        grouped_returns,
        labels=tickers,
        patch_artist=True,
        boxprops={"facecolor": "#6f8fb7", "alpha": 0.75},
        medianprops={"color": "#222222", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#b45f5f",
            "markeredgecolor": "#b45f5f",
            "markersize": 3,
            "alpha": 0.6,
        },
    )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("Daily Return Boxplot by Ticker")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Daily Return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def save_maximum_drawdown_by_ticker_chart(df: pd.DataFrame) -> Path:
    plt, FuncFormatter = get_matplotlib_pyplot()
    chart_path = EDA_CHARTS_DIR / "maximum-drawdown-by-ticker.png"
    rows: List[Dict[str, Any]] = []

    for ticker, ticker_df in df.groupby("Ticker"):
        prices = ticker_df.sort_values("Date")["Adj Close"].dropna()
        if prices.empty:
            continue
        running_peak = prices.cummax()
        drawdown = (prices / running_peak) - 1
        rows.append({
            "Ticker": ticker,
            "Maximum_Drawdown": drawdown.min(),
        })

    drawdown_df = (
        pd.DataFrame(rows)
        .sort_values("Maximum_Drawdown")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(drawdown_df["Ticker"], drawdown_df["Maximum_Drawdown"], color="#b45f5f")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Maximum Drawdown by Ticker")
    ax.set_xlabel("Maximum Drawdown")
    ax.set_ylabel("Ticker")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def generate_eda_charts(df: pd.DataFrame) -> List[Dict[str, str]]:
    chart_df = prepare_eda_chart_data(df)
    chart_specs = [
        {
            "title": "Adjusted Closing Price Trend",
            "path": save_adj_close_trend_chart(chart_df),
            "description": "Shows long-term adjusted closing price movement for each stock ticker.",
        },
        {
            "title": "Monthly Trading Volume Trend",
            "path": save_monthly_volume_trend_chart(chart_df),
            "description": "Shows monthly average trading activity while reducing daily volume noise.",
        },
        {
            "title": "Daily Return Distribution",
            "path": save_daily_return_distribution_chart(chart_df),
            "description": "Shows how daily percentage returns are distributed across the preprocessed dataset.",
        },
        {
            "title": "20-Day Rolling Volatility Trend",
            "path": save_rolling_volatility_chart(chart_df),
            "description": "Shows how short-term return volatility changes over time for each ticker.",
        },
        {
            "title": "Market Feature Correlation Heatmap",
            "path": save_correlation_heatmap_chart(chart_df),
            "description": "Shows relationships between numeric market variables in the preprocessed dataset.",
        },
        {
            "title": "Return vs Volatility by Ticker",
            "path": save_return_vs_volatility_chart(chart_df),
            "description": "Compares each stock's average daily return against its daily return volatility.",
        },
        {
            "title": "Total Return by Ticker",
            "path": save_total_return_by_ticker_chart(chart_df),
            "description": "Compares overall adjusted-close return from the first available trading date to the last available trading date.",
        },
        {
            "title": "Annual Return by Ticker",
            "path": save_annual_return_by_ticker_chart(chart_df),
            "description": "Compares year-by-year adjusted-close performance across tickers using grouped bars.",
        },
        {
            "title": "Daily Return Boxplot by Ticker",
            "path": save_daily_return_boxplot_by_ticker_chart(chart_df),
            "description": "Compares each stock's daily return spread, median, and outlier movement.",
        },
        {
            "title": "Maximum Drawdown by Ticker",
            "path": save_maximum_drawdown_by_ticker_chart(chart_df),
            "description": "Compares the worst percentage decline from a previous adjusted-close peak for each stock.",
        },
    ]

    return [
        {
            "title": chart["title"],
            "relative_path": chart["path"].relative_to(DATA_UNDERSTANDING_SUMMARY_DIR).as_posix(),
            "description": chart["description"],
        }
        for chart in chart_specs
    ]


def build_chart_markdown(chart_rows: List[Dict[str, str]]) -> str:
    if not chart_rows:
        return "_No EDA charts were generated._"

    sections: List[str] = []
    for chart in chart_rows:
        sections.append(
            f"### {chart['title']}\n"
            f"![{chart['title']}]({chart['relative_path']})\n\n"
            f"{chart['description']}"
        )

    return "\n\n".join(sections)


def write_eda_summary(df: pd.DataFrame) -> None:
    descriptive_rows = build_descriptive_rows(df)
    return_rows = build_return_rows(df)
    correlation_rows = build_correlation_rows(df)
    chart_rows = generate_eda_charts(df)
    valid_dates = pd.to_datetime(df.get("Date"), errors="coerce").dropna()
    coverage_start = str(valid_dates.min().date()) if not valid_dates.empty else "N/A"
    coverage_end = str(valid_dates.max().date()) if not valid_dates.empty else "N/A"
    split_rows = []
    if not valid_dates.empty:
        for label, start, end, purpose in (
            ("Development + purged validation", "2015-01-01", "2024-12-31", "Model and threshold selection"),
            ("Frozen observed test", "2025-01-01", "2025-12-31", "One-time acceptance evaluation"),
            ("Forward shadow", "2026-01-01", "2026-12-31", "Post-selection monitoring only"),
        ):
            count = int(valid_dates.between(pd.Timestamp(start), pd.Timestamp(end)).sum())
            split_rows.append([label, start, end, count, purpose])

    content = f"""# 3.4.2.3 Exploratory Data Analysis (EDA) Summary

## Purpose
This section summarizes the numerical behavior of the preprocessed stock dataset.

## Evidence Coverage And Evaluation Boundaries
- Observed dataset coverage: **{coverage_start} to {coverage_end}**
- The ten supported companies and KLCI context should cover at least 2015–2025
  before an official model run is considered complete.

{markdown_table(
        ["Period", "Start", "End", "Rows", "Permitted use"],
        split_rows,
    )}

The 2025 rows are frozen test evidence and must not be used for feature,
hyperparameter, calibration, or decision-threshold selection.

## EDA Visualizations
The following charts are generated from the preprocessed dataset and saved under `summary/data-understanding/eda-charts/`.

{build_chart_markdown(chart_rows)}

## Descriptive Statistics
{markdown_table(
        [
            "Column",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Q1",
            "Median",
            "Q3",
            "Max",
        ],
        descriptive_rows,
    )}

## Daily Return Summary By Ticker
{markdown_table(
        [
            "Ticker",
            "Return Count",
            "Mean Return",
            "Return Std",
            "Minimum Return",
            "Maximum Return",
        ],
        return_rows,
    )}

## Highest Absolute Correlations
{markdown_table(
        [
            "Column A",
            "Column B",
            "Correlation",
        ],
        correlation_rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "eda-summary.md", content)


# ============================================================
# 3.4.2.4 Dataset Insights
# ============================================================
def write_dataset_insights_summary(df: pd.DataFrame) -> None:
    row_count = len(df)
    ticker_count = int(df["Ticker"].nunique(dropna=True)) if "Ticker" in df.columns else 0
    first_date = df["Date"].min().date() if not df["Date"].dropna().empty else "N/A"
    last_date = df["Date"].max().date() if not df["Date"].dropna().empty else "N/A"
    missing_total = int(df.drop(columns=["Source_File"], errors="ignore").isna().sum().sum())

    if MODEL_INPUT_SUMMARY_FILE.exists():
        model_input_summary_df = pd.read_csv(MODEL_INPUT_SUMMARY_FILE)
        model_input_rows = int(model_input_summary_df["final_rows"].sum())
        model_input_files = int(model_input_summary_df["file_name"].nunique())
        model_input_note = (
            f"The model input-ready dataset currently contains {model_input_rows:,} "
            f"rows across {model_input_files} files according to "
            "`data/model-input-summary.csv`."
        )
    elif get_model_input_csv_files():
        model_input_files = get_model_input_csv_files()
        model_input_note = (
            f"The model input-ready dataset exists across {len(model_input_files)} "
            "files, but `data/model-input-summary.csv` was not found."
        )
    else:
        model_input_note = (
            "The model input-ready dataset has not been generated yet. Run "
            "`python main.py build_model_input` after preprocessing to create it."
        )

    volume = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df.columns else pd.Series(dtype=float)

    volume_note = "Volume data is unavailable."
    if not volume.dropna().empty:
        volume_note = (
            f"Trading volume ranges from {volume.min():,.0f} to {volume.max():,.0f}, "
            "which means volume-related features can be much larger in scale than price features."
        )

    close_note = "Closing price data is unavailable."
    if not close.dropna().empty:
        close_note = (
            f"Closing prices range from {close.min():,.4f} to {close.max():,.4f} "
            "in the preprocessed dataset."
        )

    content = f"""# 3.4.2.4 Dataset Insights Summary

## Purpose
This section interprets the preprocessed dataset quality and explains the modeling implications of the observed data.

## Dataset Quality
- The preprocessed dataset contains {row_count:,} rows across {ticker_count} tickers.
- The date range is {first_date} to {last_date}.
- Total missing values remaining in the preprocessed dataset: {missing_total:,}.
- The preprocessing stage keeps statistical market outliers because unusual price or volume movements may contain useful prediction signals.

## Market Data Insights
- {close_note}
- {volume_note}
- Because price and volume features have very different magnitudes, scaling is useful for scale-sensitive models such as Logistic Regression.
- Tree-based models such as Random Forest and XGBoost can usually work without standardization because their splits are threshold-based.

## Modeling Implications
- `Adj Close` is important because it accounts for corporate actions and is used for return, lag, moving-average, volatility, and target construction.
- Time order should be preserved during modeling to avoid future-data leakage.
- Scaling should be fitted only on the training or development period, then applied to later validation or test periods.
- {model_input_note}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "dataset-insights-summary.md", content)


# ============================================================
# Main pipeline
# ============================================================
def run_data_understanding_pipeline() -> None:
    ensure_summary_structure()

    preprocessed_files = get_preprocessed_csv_files()
    if not preprocessed_files:
        print(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")
        return

    print("[Starting data understanding summary pipeline]\n")

    df = load_stock_files(preprocessed_files)

    write_variables_summary(df)
    write_observations_summary(df)
    write_eda_summary(df)
    write_dataset_insights_summary(df)

    print(f"Data understanding summaries saved to: {DATA_UNDERSTANDING_SUMMARY_DIR}")
    print("\n[Data understanding summary pipeline finished]")


def main() -> None:
    run_data_understanding_pipeline()


if __name__ == "__main__":
    main()
