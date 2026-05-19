"""Streamlit dashboard for the sales forecasting pipeline.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_DATA_PATH = Path("data/engineered_sales.csv")


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """Load engineered sales data for dashboard visualizations.

    Args:
        csv_path: Path to an engineered sales CSV.

    Returns:
        Parsed sales data sorted by store and date.
    """

    data = pd.read_csv(csv_path)
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    return data.sort_values(["Store", "Date"]).reset_index(drop=True)


def format_currency(value: float) -> str:
    """Format a numeric value as a compact currency-like sales figure."""

    return f"{value:,.0f}"


def metric_delta(current: float, previous: float) -> str:
    """Return a readable percentage delta between two values."""

    if previous == 0:
        return "n/a"
    return f"{((current - previous) / previous) * 100:.1f}%"


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""

    st.sidebar.header("Filters")

    stores = sorted(data["Store"].unique().tolist())
    selected_stores = st.sidebar.multiselect("Store", stores, default=stores)

    min_date = data["Date"].min().date()
    max_date = data["Date"].max().date()
    selected_dates = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    promo_options = ["All", "Promo only", "No promo"]
    promo_filter = st.sidebar.radio("Promotion", promo_options, horizontal=False)

    filtered = data[data["Store"].isin(selected_stores)]

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered = filtered[
            (filtered["Date"].dt.date >= start_date)
            & (filtered["Date"].dt.date <= end_date)
        ]

    if promo_filter == "Promo only":
        filtered = filtered[filtered["Promo"] == 1]
    elif promo_filter == "No promo":
        filtered = filtered[filtered["Promo"] == 0]

    return filtered


def render_kpis(data: pd.DataFrame) -> None:
    """Render headline business metrics."""

    total_sales = data["Sales"].sum()
    total_customers = data["Customers"].sum()
    avg_sales = data["Sales"].mean()
    avg_basket = total_sales / total_customers if total_customers else 0

    daily_sales = data.groupby("Date", as_index=False)["Sales"].sum().sort_values("Date")
    latest_sales = daily_sales["Sales"].iloc[-1] if not daily_sales.empty else 0
    previous_sales = daily_sales["Sales"].iloc[-2] if len(daily_sales) > 1 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", format_currency(total_sales))
    col2.metric("Customers", format_currency(total_customers))
    col3.metric("Avg Daily Sales", format_currency(avg_sales))
    col4.metric("Latest Day Sales", format_currency(latest_sales), metric_delta(latest_sales, previous_sales))

    st.caption(f"Estimated sales per customer: {avg_basket:.2f}")


def render_sales_trend(data: pd.DataFrame) -> None:
    """Render sales trend chart with optional rolling average."""

    daily = data.groupby("Date", as_index=False).agg(Sales=("Sales", "sum"), Customers=("Customers", "sum"))
    daily["rolling_7d"] = daily["Sales"].rolling(7, min_periods=1).mean()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=daily["Date"], y=daily["Sales"], mode="lines+markers", name="Actual sales"))
    figure.add_trace(go.Scatter(x=daily["Date"], y=daily["rolling_7d"], mode="lines", name="7-day trend"))
    figure.update_layout(
        title="Sales Trend",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_store_comparison(data: pd.DataFrame) -> None:
    """Render store-level comparison charts."""

    store_summary = (
        data.groupby("Store", as_index=False)
        .agg(Sales=("Sales", "sum"), Customers=("Customers", "sum"))
        .sort_values("Sales", ascending=False)
    )
    store_summary["sales_per_customer"] = store_summary["Sales"] / store_summary["Customers"].replace(0, pd.NA)

    figure = px.bar(
        store_summary,
        x="Store",
        y="Sales",
        color="sales_per_customer",
        title="Sales by Store",
        labels={"sales_per_customer": "Sales / Customer"},
    )
    figure.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(figure, use_container_width=True)


def render_business_insights(data: pd.DataFrame) -> None:
    """Render concise insight panels derived from the filtered data."""

    promo_sales = data.groupby("Promo")["Sales"].mean().to_dict()
    promo_lift = None
    if 0 in promo_sales and 1 in promo_sales and promo_sales[0] != 0:
        promo_lift = ((promo_sales[1] - promo_sales[0]) / promo_sales[0]) * 100

    weekday = data.groupby("day_of_week", as_index=False)["Sales"].mean()
    weekday_name = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    best_day = None
    if not weekday.empty:
        best_row = weekday.loc[weekday["Sales"].idxmax()]
        best_day = weekday_name.get(int(best_row["day_of_week"]), str(best_row["day_of_week"]))

    col1, col2, col3 = st.columns(3)
    col1.subheader("Promotion Effect")
    col1.write(f"{promo_lift:.1f}% average sales lift" if promo_lift is not None else "Need promo and non-promo days to compare.")

    col2.subheader("Best Sales Day")
    col2.write(best_day or "Not enough data yet.")

    col3.subheader("Open Days")
    open_rate = data["Open"].mean() * 100 if "Open" in data else 0
    col3.write(f"{open_rate:.1f}% of selected rows were open.")


def render_feature_view(data: pd.DataFrame) -> None:
    """Render engineered feature preview and missing value summary."""

    feature_columns = [
        "Sales_lag_7",
        "Sales_lag_14",
        "Sales_lag_30",
        "Sales_roll_mean_7",
        "Sales_roll_mean_14",
        "Sales_roll_mean_30",
        "is_state_holiday",
        "is_weekend",
    ]
    visible_columns = ["Date", "Store", "Sales", "Customers", "Promo"] + [
        column for column in feature_columns if column in data.columns
    ]

    st.dataframe(data[visible_columns], use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit dashboard."""

    st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
    st.title("Real-Time Sales Forecasting Dashboard")
    st.caption("Pipeline view for sales trends, feature signals, and early business insights.")

    csv_path = st.sidebar.text_input("Engineered CSV path", value=str(DEFAULT_DATA_PATH))

    if not Path(csv_path).exists():
        st.error(f"Could not find {csv_path}. Run `python run_pipeline.py` first.")
        st.stop()

    data = load_data(csv_path)
    filtered = filter_data(data)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        st.stop()

    render_kpis(filtered)

    tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Stores", "Insights", "Features"])
    with tab1:
        render_sales_trend(filtered)
    with tab2:
        render_store_comparison(filtered)
    with tab3:
        render_business_insights(filtered)
    with tab4:
        render_feature_view(filtered)


if __name__ == "__main__":
    main()
