from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(
    page_title="Air Passenger Forecast",
    page_icon="✈️",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "airline-passengers.csv"


@st.cache_resource
def fit_forecast(values: tuple[float, ...], first_month: str, horizon: int):
    dates = pd.date_range(start=first_month, periods=len(values), freq="MS")
    series = pd.Series(values, index=dates, name="Passengers")
    fitted_model = ARIMA(series, order=(2, 1, 2)).fit()
    forecast_result = fitted_model.get_forecast(steps=horizon)

    future_dates = pd.date_range(
        start=dates[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    forecast = pd.Series(
        forecast_result.predicted_mean.to_numpy(),
        index=future_dates,
        name="Forecast",
    )
    interval = forecast_result.conf_int()
    interval.index = future_dates
    return series, forecast, interval


def prepare_data(frame: pd.DataFrame) -> pd.Series:
    required_columns = {"Month", "Passengers"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("The CSV must contain Month and Passengers columns.")

    frame = frame[["Month", "Passengers"]].copy()
    frame["Month"] = pd.to_datetime(frame["Month"], errors="coerce")
    frame["Passengers"] = pd.to_numeric(frame["Passengers"], errors="coerce")
    frame = frame.dropna().sort_values("Month")
    frame["Month"] = frame["Month"].dt.to_period("M").dt.to_timestamp()

    if frame.empty:
        raise ValueError("The CSV does not contain valid observations.")
    if frame["Month"].duplicated().any():
        raise ValueError("There must be only one observation per month.")
    if len(frame) < 24:
        raise ValueError("Please provide at least 24 consecutive monthly observations.")

    series = frame.set_index("Month")["Passengers"].asfreq("MS")
    if series.isna().any():
        raise ValueError("The time series has missing months. Fill those months before forecasting.")
    return series


st.title("Air Passenger Forecast")
st.write("Explore monthly passenger totals and generate forecasts with ARIMA(2, 1, 2).")

with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Months to forecast", min_value=1, max_value=36, value=24)
    uploaded_file = st.file_uploader(
        "Use another monthly CSV (optional)",
        type="csv",
        help="CSV columns must be named Month and Passengers.",
    )

try:
    if uploaded_file is not None:
        source_data = pd.read_csv(uploaded_file)
    else:
        source_data = pd.read_csv(DATA_PATH)

    observations = prepare_data(source_data)
    history, forecast, confidence_interval = fit_forecast(
        tuple(float(value) for value in observations),
        observations.index[0].strftime("%Y-%m-%d"),
        horizon,
    )
except FileNotFoundError:
    st.error("The bundled airline-passengers.csv file is missing.")
    st.stop()
except (ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"The forecast could not be generated: {error}")
    st.stop()

metric_col, detail_col = st.columns(2)
metric_col.metric("Latest observed value", f"{history.iloc[-1]:,.0f} thousand")
detail_col.metric("Forecast period", f"{horizon} months")

figure, axis = plt.subplots(figsize=(12, 5))
axis.plot(history.index, history, color="#176B87", linewidth=1.8, label="Historical")
axis.plot(forecast.index, forecast, color="#D95D39", linewidth=2, label="Forecast")
axis.fill_between(
    confidence_interval.index,
    confidence_interval.iloc[:, 0],
    confidence_interval.iloc[:, 1],
    color="#D95D39",
    alpha=0.18,
    label="95% confidence interval",
)
axis.axvline(history.index[-1], color="#65727A", linestyle=":", linewidth=1.2)
axis.set_xlabel("Month")
axis.set_ylabel("Passengers (thousands)")
axis.grid(alpha=0.22)
axis.legend(frameon=False, ncol=3)
figure.tight_layout()
st.pyplot(figure, use_container_width=True)
plt.close(figure)

st.subheader("Forecast values")
forecast_table = forecast.to_frame().round(1)
st.dataframe(forecast_table, use_container_width=True)
st.download_button(
    "Download forecast CSV",
    data=forecast_table.to_csv(index_label="Month").encode("utf-8"),
    file_name="air-passenger-forecast.csv",
    mime="text/csv",
)

st.caption(
    f"Model: ARIMA(2, 1, 2) · Trained on {len(history)} monthly observations "
    f"from {history.index[0]:%b %Y} to {history.index[-1]:%b %Y}."
)
