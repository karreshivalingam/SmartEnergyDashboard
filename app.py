# =========================================================
# ⚡ Smart Energy Forecasting & Optimization — Germany
# Phase 7 Streamlit Dashboard (Advanced)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import joblib
from datetime import date, timedelta
from prophet import Prophet

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Smart Energy Forecasting & Optimization — Germany",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Energy Forecasting & Optimization — Germany")
st.caption("Phase 7 Dashboard: Historical + Forecasting (RF+API / Prophet+Climatology) + Optimization + Model Insights")

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/Merged_SMARD_Weather_Daily.csv"
RF_PATH = "models/rf_model.joblib"

# =========================================================
# 1) Load data & model
# =========================================================
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_resource
def load_rf(path=RF_PATH):
    return joblib.load(path)

data = load_data()
rf_model = load_rf()

# Feature columns (prefer model feature names if available)
if hasattr(rf_model, "feature_names_in_"):
    feature_cols = list(rf_model.feature_names_in_)
else:
    feature_cols = [c for c in data.columns if c not in ["Date", "Grid_Load_MWh"]]

# Weather cols expected in your dataset
weather_cols = ["Temperature_F", "Humidity_Percent", "SolarRadiation_Wm2", "Precipitation_mm", "WindSpeed_kmh"]

# Ensure calendar cols exist
if "Month" not in data.columns:
    data["Month"] = data["Date"].dt.month
if "DayOfWeek" not in data.columns:
    data["DayOfWeek"] = data["Date"].dt.dayofweek
if "IsWeekend" not in data.columns:
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)

# Historical range
hist_start = data["Date"].min().date()
hist_end = data["Date"].max().date()

# =========================================================
# 2) Prophet model (cached)
# =========================================================
@st.cache_resource
def fit_prophet(df):
    prophet_df = df[["Date", "Grid_Load_MWh"]].rename(columns={"Date": "ds", "Grid_Load_MWh": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(prophet_df)
    return m

prophet_model = fit_prophet(data)

# =========================================================
# 3) Climatology (typical weather by Month + DayOfWeek)
# =========================================================
@st.cache_data
def build_climatology(df):
    tmp = df.copy()
    tmp["Month"] = tmp["Date"].dt.month
    tmp["DayOfWeek"] = tmp["Date"].dt.dayofweek
    return tmp.groupby(["Month", "DayOfWeek"])[weather_cols].mean().reset_index()

climatology = build_climatology(data)

def get_typical_weather_for_date(dt: pd.Timestamp):
    month = dt.month
    dow = dt.weekday()
    row = climatology[(climatology["Month"] == month) & (climatology["DayOfWeek"] == dow)]
    if not row.empty:
        return row.iloc[0][weather_cols]
    row = climatology[climatology["Month"] == month]
    if not row.empty:
        return row[weather_cols].mean()
    return climatology[weather_cols].mean()

def attach_climatology_weather(dates: pd.Series):
    records = []
    for d in pd.to_datetime(dates):
        typical = get_typical_weather_for_date(d)
        rec = {"Date": d.date()}
        for c in weather_cols:
            rec[c] = float(typical[c])
        records.append(rec)
    return pd.DataFrame(records)

# =========================================================
# 4) Historical data check
# =========================================================
def get_actual_data_if_available(start_date: date, end_date: date):
    if start_date >= hist_start and end_date <= hist_end:
        mask = (data["Date"].dt.date >= start_date) & (data["Date"].dt.date <= end_date)
        cols = ["Date", "Grid_Load_MWh"] + [c for c in weather_cols if c in data.columns]
        return data.loc[mask, cols].copy()
    return pd.DataFrame()

# =========================================================
# 5) Open-Meteo SHORT RANGE Weather API (next 16 days)
# =========================================================
def get_weather_api_forecast_short_range():
    latitude = 51.1657
    longitude = 10.4515

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "shortwave_radiation_sum",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]),
        "forecast_days": 16,
        "timezone": "Europe/Berlin",
    }

    url = "https://api.open-meteo.com/v1/forecast"
    resp = requests.get(url, params=params, timeout=25)
    resp.raise_for_status()

    js = resp.json()
    daily = js.get("daily", {})
    if "time" not in daily:
        return pd.DataFrame()

    df_weather = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Temperature_F": (pd.Series(daily["temperature_2m_mean"]) * 9 / 5) + 32,
        "Humidity_Percent": daily["relative_humidity_2m_mean"],
        "SolarRadiation_Wm2": pd.Series(daily["shortwave_radiation_sum"]) / 24.0,
        "Precipitation_mm": daily["precipitation_sum"],
        "WindSpeed_kmh": daily["wind_speed_10m_max"],
    }).sort_values("Date").reset_index(drop=True)

    return df_weather

def api_window(df_weather: pd.DataFrame):
    if df_weather.empty:
        return None, None
    return df_weather["Date"].min().date(), df_weather["Date"].max().date()

@st.cache_data(ttl=60 * 30)
def cached_api_weather():
    return get_weather_api_forecast_short_range()

# =========================================================
# 6) Short-term forecasting: Random Forest + API weather (RECURSIVE)
#    + optional what-if scenario adjustments
# =========================================================
def forecast_short_term_rf_api(start_date: date, end_date: date, max_days=14,
                              temp_delta_f: float = 0.0, solar_multiplier: float = 1.0):
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    horizon_days = (end_ts - start_ts).days + 1
    if horizon_days <= 0 or horizon_days > max_days:
        return pd.DataFrame(), f"Short-term horizon must be 1..{max_days} days. Requested: {horizon_days}."

    try:
        api_weather = cached_api_weather()
    except Exception as e:
        return pd.DataFrame(), f"Weather API error: {e}"

    if api_weather.empty:
        return pd.DataFrame(), "Weather API returned empty data."

    mask = (api_weather["Date"] >= start_ts) & (api_weather["Date"] <= end_ts)
    weather_future = api_weather.loc[mask].copy().sort_values("Date").reset_index(drop=True)

    if weather_future.empty:
        api_min, api_max = api_window(api_weather)
        return pd.DataFrame(), f"Requested dates are outside the API forecast window ({api_min} → {api_max})."

    # Apply what-if adjustments (optional)
    weather_future["Temperature_F"] = weather_future["Temperature_F"] + float(temp_delta_f)
    weather_future["SolarRadiation_Wm2"] = weather_future["SolarRadiation_Wm2"] * float(solar_multiplier)

    history = data.sort_values("Date").copy().reset_index(drop=True)
    forecasts = []

    for i in range(len(weather_future)):
        row = weather_future.iloc[i]
        next_date = row["Date"]

        last_row = history.iloc[-1].copy()
        last_row["Date"] = next_date

        # Calendar
        if "Month" in history.columns:
            last_row["Month"] = next_date.month
        if "DayOfWeek" in history.columns:
            last_row["DayOfWeek"] = next_date.weekday()
        if "IsWeekend" in history.columns:
            last_row["IsWeekend"] = int(next_date.weekday() >= 5)

        # Weather from API
        for col in weather_cols:
            if col in history.columns and col in weather_future.columns:
                last_row[col] = row[col]

        # Lag / MA features if present
        recent = history["Grid_Load_MWh"].values
        if "Lag_1" in history.columns:
            last_row["Lag_1"] = recent[-1]
        if "Lag_7" in history.columns:
            last_row["Lag_7"] = recent[-7] if len(recent) >= 7 else recent[-1]
        if "Lag_30" in history.columns:
            last_row["Lag_30"] = recent[-30] if len(recent) >= 30 else recent[0]
        if "MA_7" in history.columns:
            last_row["MA_7"] = history["Grid_Load_MWh"].tail(7).mean()
        if "MA_30" in history.columns:
            last_row["MA_30"] = history["Grid_Load_MWh"].tail(30).mean()

        # Temperature categories if present
        if "Temperature_F" in history.columns:
            temp_f = float(last_row["Temperature_F"])
            if "Cold_Day" in history.columns:
                last_row["Cold_Day"] = int(temp_f < 40)
            if "Mild_Day" in history.columns:
                last_row["Mild_Day"] = int(40 <= temp_f <= 70)
            if "Hot_Day" in history.columns:
                last_row["Hot_Day"] = int(temp_f > 70)

        # Interaction features if present
        if "Temp_Solar" in history.columns and "SolarRadiation_Wm2" in history.columns:
            last_row["Temp_Solar"] = float(last_row["Temperature_F"]) * float(last_row["SolarRadiation_Wm2"])
        if "Temp_Humidity" in history.columns and "Humidity_Percent" in history.columns:
            last_row["Temp_Humidity"] = float(last_row["Temperature_F"]) * float(last_row["Humidity_Percent"])
        if "TempSquared" in history.columns and "Temperature_F" in history.columns:
            last_row["TempSquared"] = float(last_row["Temperature_F"]) ** 2

        # Predict
        X_row = last_row.reindex(feature_cols).to_frame().T
        X_row = X_row.fillna(0)

        y_hat = float(rf_model.predict(X_row)[0])

        forecasts.append({
            "Date": next_date.date(),
            "Predicted_Load_MW": y_hat,
            "Temperature_F": float(row["Temperature_F"]),
            "Humidity_Percent": float(row["Humidity_Percent"]),
            "SolarRadiation_Wm2": float(row["SolarRadiation_Wm2"]),
            "Precipitation_mm": float(row["Precipitation_mm"]),
            "WindSpeed_kmh": float(row["WindSpeed_kmh"]),
        })

        # Append prediction to history
        last_row["Grid_Load_MWh"] = y_hat
        history = pd.concat([history, last_row.to_frame().T], ignore_index=True)

    return pd.DataFrame(forecasts), None

# =========================================================
# 7) Long-term forecasting: Prophet + climatology weather
# =========================================================
def forecast_long_term_prophet(start_date: date, end_date: date):
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    future_dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
    if len(future_dates) == 0:
        return pd.DataFrame()

    future_df = pd.DataFrame({"ds": future_dates})
    fc = prophet_model.predict(future_df)

    out = fc.rename(columns={
        "ds": "Date",
        "yhat": "Expected_Load_MW",
        "yhat_lower": "Min_Expected_Load_MW",
        "yhat_upper": "Max_Expected_Load_MW"
    })[["Date", "Expected_Load_MW", "Min_Expected_Load_MW", "Max_Expected_Load_MW"]].copy()

    out["Date"] = pd.to_datetime(out["Date"])
    w = attach_climatology_weather(out["Date"])
    out["Date"] = out["Date"].dt.date
    return out.merge(w, on="Date", how="left")

# =========================================================
# 8) Phase 6 Optimization (load shifting)
# =========================================================
def run_optimization(forecast_df: pd.DataFrame, flexible_fraction=0.10, normal_price=120, peak_price=180):
    df = forecast_df.copy()

    if "Expected_Load_MW" in df.columns:
        load_col = "Expected_Load_MW"
    elif "Predicted_Load_MW" in df.columns:
        load_col = "Predicted_Load_MW"
    else:
        return pd.DataFrame(), {}

    # Guardrail: optimization needs multiple days to shift
    if len(df) < 2:
        df["Energy_MWh"] = df[load_col] * 24.0
        df["IsPeakDay"] = False
        df["Price_EUR_per_MWh"] = normal_price
        df["Baseline_Cost_EUR"] = df["Energy_MWh"] * df["Price_EUR_per_MWh"]
        df["Flexible_Energy_MWh"] = 0.0
        df["Optimized_Energy_MWh"] = df["Energy_MWh"]
        df["Optimized_Load_MW"] = df[load_col]
        df["Optimized_Cost_EUR"] = df["Baseline_Cost_EUR"]
        summary = {
            "baseline_total_cost_EUR": float(df["Baseline_Cost_EUR"].sum()),
            "optimized_total_cost_EUR": float(df["Baseline_Cost_EUR"].sum()),
            "absolute_savings_EUR": 0.0,
            "relative_savings_percent": 0.0,
            "flexible_fraction": flexible_fraction,
            "n_days": int(len(df)),
            "n_expensive_days": 0,
            "n_cheap_days": 0
        }
        return df, summary

    df["Energy_MWh"] = df[load_col] * 24.0

    threshold = df[load_col].quantile(0.70)
    df["IsPeakDay"] = df[load_col] >= threshold
    df["Price_EUR_per_MWh"] = np.where(df["IsPeakDay"], peak_price, normal_price)
    df["Baseline_Cost_EUR"] = df["Energy_MWh"] * df["Price_EUR_per_MWh"]

    baseline_total_cost = float(df["Baseline_Cost_EUR"].sum())

    df_sorted = df.sort_values("Baseline_Cost_EUR").reset_index(drop=True)
    n = len(df_sorted)
    n_cheap = max(1, int(0.25 * n))
    n_expensive = max(1, int(0.25 * n))

    cheap_idx = df_sorted.index[:n_cheap]
    expensive_idx = df_sorted.index[-n_expensive:]

    df_sorted["Flexible_Energy_MWh"] = df_sorted["Energy_MWh"] * flexible_fraction

    shift_energy = float(df_sorted.loc[expensive_idx, "Flexible_Energy_MWh"].sum())

    df_sorted["Optimized_Energy_MWh"] = df_sorted["Energy_MWh"]
    df_sorted.loc[expensive_idx, "Optimized_Energy_MWh"] = (
        df_sorted.loc[expensive_idx, "Energy_MWh"] - df_sorted.loc[expensive_idx, "Flexible_Energy_MWh"]
    )

    add_each = shift_energy / len(cheap_idx)
    df_sorted.loc[cheap_idx, "Optimized_Energy_MWh"] = df_sorted.loc[cheap_idx, "Energy_MWh"] + add_each

    df_sorted["Optimized_Load_MW"] = df_sorted["Optimized_Energy_MWh"] / 24.0
    df_sorted["Optimized_Cost_EUR"] = df_sorted["Optimized_Energy_MWh"] * df_sorted["Price_EUR_per_MWh"]

    optimized_total_cost = float(df_sorted["Optimized_Cost_EUR"].sum())
    absolute_savings = baseline_total_cost - optimized_total_cost
    relative_savings = (absolute_savings / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0.0

    df_out = df_sorted.sort_values("Date").reset_index(drop=True)

    summary = {
        "baseline_total_cost_EUR": baseline_total_cost,
        "optimized_total_cost_EUR": optimized_total_cost,
        "absolute_savings_EUR": absolute_savings,
        "relative_savings_percent": relative_savings,
        "flexible_fraction": flexible_fraction,
        "n_days": int(n),
        "n_expensive_days": int(n_expensive),
        "n_cheap_days": int(n_cheap),
    }
    return df_out, summary

# =========================================================
# 9) Sidebar controls (date selection + optimization settings)
# =========================================================
st.sidebar.header("Controls")

mode = st.sidebar.radio("Select mode", ["Single date", "Date range", "Full year"], index=0)

default_end = hist_end
default_start = (pd.to_datetime(default_end) - pd.Timedelta(days=7)).date()

if mode == "Single date":
    d = st.sidebar.date_input("Select date", value=default_end)
    start_date = d
    end_date = d
elif mode == "Date range":
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=default_end)
else:
    y = st.sidebar.number_input("Year", min_value=2015, max_value=2035, value=2025, step=1)
    start_date = date(int(y), 1, 1)
    end_date = date(int(y), 12, 31)

if start_date > end_date:
    st.sidebar.error("Start date must be <= end date.")
    st.stop()

st.sidebar.subheader("Optimization parameters")
flexible_fraction = st.sidebar.slider("Flexible fraction", 0.05, 0.30, 0.10, 0.01)
normal_price = st.sidebar.number_input("Normal price (EUR/MWh)", value=120, step=5)
peak_price = st.sidebar.number_input("Peak price (EUR/MWh)", value=180, step=5)

st.sidebar.subheader("What-if scenario (short-term RF only)")
temp_delta_f = st.sidebar.slider("Temperature adjustment (°F)", -10.0, 10.0, 0.0, 0.5)
solar_multiplier = st.sidebar.slider("Solar radiation multiplier", 0.7, 1.3, 1.0, 0.05)

# =========================================================
# 10) Tabs
# =========================================================
tab_overview, tab_hist, tab_fc, tab_opt, tab_models = st.tabs(
    ["📌 Overview", "📊 Historical", "🔮 Forecasting", "⚙ Optimization", "📈 Model Comparison"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset start", str(hist_start))
    c2.metric("Dataset end", str(hist_end))
    c3.metric("Total days", f"{len(data):,}")
    c4.metric("Avg daily load", f"{data['Grid_Load_MWh'].mean():,.0f}")

    st.write("### Project Flow (Automatic Selection)")
    st.write(
        "- If selected dates exist in the dataset → show **actual historical values**.\n"
        "- Else if selected range is short (≤ 14 days) and inside the **API forecast window** → forecast using **Random Forest + live weather API**.\n"
        "- Otherwise → forecast using **Prophet + climatology weather** (long-term).\n"
    )

    # Show API window (if available)
    try:
        api_weather = cached_api_weather()
        api_min, api_max = api_window(api_weather)
        if api_min and api_max:
            st.info(f"Live Weather API Window (today’s forecast): **{api_min} → {api_max}**")
        else:
            st.warning("Live Weather API returned no window right now.")
    except Exception as e:
        st.warning(f"Live Weather API could not be loaded: {e}")

    st.write("**Units note:** Load values are displayed as **MW** in this dashboard (consistent with project interpretation).")

# -----------------------------
# Historical
# -----------------------------
with tab_hist:
    st.subheader("Historical Data (Actual values if available)")
    actual = get_actual_data_if_available(start_date, end_date)

    if actual.empty:
        st.info("Selected dates are not fully inside the historical dataset (actual values not available for entire range).")
    else:
        st.success("Actual historical data found for the selected date range.")
        show = actual.copy()
        show["Date"] = pd.to_datetime(show["Date"]).dt.date

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg load (MW)", f"{show['Grid_Load_MWh'].mean():,.0f}")
        k2.metric("Max load (MW)", f"{show['Grid_Load_MWh'].max():,.0f}")
        k3.metric("Min load (MW)", f"{show['Grid_Load_MWh'].min():,.0f}")

        st.dataframe(show.style.format({
            "Grid_Load_MWh": "{:,.0f}",
            "Temperature_F": "{:,.2f}",
            "Humidity_Percent": "{:,.2f}",
            "SolarRadiation_Wm2": "{:,.2f}",
            "Precipitation_mm": "{:,.2f}",
            "WindSpeed_kmh": "{:,.2f}",
        }), use_container_width=True)

        # Download
        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download historical data (CSV)", csv, "historical_data.csv", "text/csv")

        fig = plt.figure(figsize=(12, 4))
        plt.plot(pd.to_datetime(actual["Date"]), actual["Grid_Load_MWh"])
        plt.title("Historical Electricity Consumption (MW)")
        plt.xlabel("Date")
        plt.ylabel("Load (MW)")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

# -----------------------------
# Forecasting (Automatic Flow)
# -----------------------------
with tab_fc:
    st.subheader("Forecasting (Automatic Flow)")

    actual = get_actual_data_if_available(start_date, end_date)
    horizon_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    short_term_max_days = 14

    # Explain selection box
    with st.expander("Why this forecasting method is chosen (click to expand)", expanded=True):
        st.write(f"- Selected range: **{start_date} → {end_date}** (**{horizon_days} day(s)**)")
        st.write(f"- Historical dataset window: **{hist_start} → {hist_end}**")

        try:
            api_weather = cached_api_weather()
            api_min, api_max = api_window(api_weather)
            st.write(f"- Live weather API window: **{api_min} → {api_max}**")
        except Exception:
            api_min, api_max = None, None
            st.write("- Live weather API window: **not available (API error)**")

        if not actual.empty:
            st.write("✅ **Method: Historical (actual values)** because selected dates are fully inside dataset.")
        elif horizon_days <= short_term_max_days and api_min and api_max and (start_date >= api_min and end_date <= api_max):
            st.write("✅ **Method: Short-term RF + API** because range is short and inside API forecast window.")
        else:
            if horizon_days > short_term_max_days:
                st.write("✅ **Method: Prophet + climatology** because the selected range is longer than short-term horizon.")
            else:
                st.write("✅ **Method: Prophet + climatology** because the selected dates are outside the API forecast window or API is not available.")

    if not actual.empty:
        st.success("✅ Selected dates are available in the dataset → showing ACTUAL values (no forecasting needed).")
        show = actual.copy()
        show["Date"] = pd.to_datetime(show["Date"]).dt.date
        st.dataframe(show.style.format({"Grid_Load_MWh": "{:,.0f}"}), use_container_width=True)

    else:
        # Try RF+API only if short horizon
        st.write("### Short-term Forecast (Random Forest + Weather API)")
        api_min, api_max = None, None
        try:
            api_weather = cached_api_weather()
            api_min, api_max = api_window(api_weather)
            if api_min and api_max:
                st.caption(f"Weather API forecast window available right now: {api_min} → {api_max}")
        except Exception as e:
            st.error(f"Weather API error: {e}")

        use_rf_api = False
        if horizon_days <= short_term_max_days and api_min and api_max:
            if start_date >= api_min and end_date <= api_max:
                use_rf_api = True

        if use_rf_api:
            rf_fc, rf_err = forecast_short_term_rf_api(
                start_date, end_date,
                max_days=short_term_max_days,
                temp_delta_f=temp_delta_f,
                solar_multiplier=solar_multiplier
            )
            if rf_err is None and not rf_fc.empty:
                st.success("✅ SHORT-TERM: Random Forest + live API weather")

                # KPI cards
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Avg forecast (MW)", f"{rf_fc['Predicted_Load_MW'].mean():,.0f}")
                k2.metric("Max forecast (MW)", f"{rf_fc['Predicted_Load_MW'].max():,.0f}")
                k3.metric("Min forecast (MW)", f"{rf_fc['Predicted_Load_MW'].min():,.0f}")
                k4.metric("Horizon (days)", f"{len(rf_fc)}")

                st.dataframe(rf_fc.style.format({
                    "Predicted_Load_MW": "{:,.0f}",
                    "Temperature_F": "{:,.2f}",
                    "Humidity_Percent": "{:,.2f}",
                    "SolarRadiation_Wm2": "{:,.2f}",
                    "Precipitation_mm": "{:,.2f}",
                    "WindSpeed_kmh": "{:,.2f}",
                }), use_container_width=True)

                # Download
                csv = rf_fc.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download RF forecast (CSV)", csv, "rf_forecast.csv", "text/csv")

                fig = plt.figure(figsize=(12, 4))
                plt.plot(pd.to_datetime(rf_fc["Date"]), rf_fc["Predicted_Load_MW"])
                plt.title("Short-term Forecast (Random Forest + API weather)")
                plt.xlabel("Date")
                plt.ylabel("Predicted Load (MW)")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.markdown(f"**⚠ API/RF issue:** {rf_err}")
                st.markdown("**Switching to LONG-TERM forecasting (Prophet + climatology).**")

                lt = forecast_long_term_prophet(start_date, end_date)
                st.success("✅ LONG-TERM: Prophet + climatology weather")
                st.dataframe(lt.style.format({
                    "Expected_Load_MW": "{:,.0f}",
                    "Min_Expected_Load_MW": "{:,.0f}",
                    "Max_Expected_Load_MW": "{:,.0f}",
                }), use_container_width=True)

        else:
            # Show bold out-of-range warning if needed
            if horizon_days <= short_term_max_days:
                st.markdown("**⚠ Selected range is not inside the Weather API forecast window → using Prophet.**")
            else:
                st.markdown(f"**⚠ Requested horizon = {horizon_days} days (> {short_term_max_days}) → using Prophet.**")

            lt = forecast_long_term_prophet(start_date, end_date)
            st.success("✅ LONG-TERM: Prophet + climatology weather")

            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg expected (MW)", f"{lt['Expected_Load_MW'].mean():,.0f}")
            k2.metric("Max expected (MW)", f"{lt['Expected_Load_MW'].max():,.0f}")
            k3.metric("Min expected (MW)", f"{lt['Expected_Load_MW'].min():,.0f}")
            k4.metric("Horizon (days)", f"{len(lt)}")

            st.dataframe(lt.style.format({
                "Expected_Load_MW": "{:,.0f}",
                "Min_Expected_Load_MW": "{:,.0f}",
                "Max_Expected_Load_MW": "{:,.0f}",
                "Temperature_F": "{:,.2f}",
                "Humidity_Percent": "{:,.2f}",
                "SolarRadiation_Wm2": "{:,.2f}",
                "Precipitation_mm": "{:,.2f}",
                "WindSpeed_kmh": "{:,.2f}",
            }), use_container_width=True)

            # Download
            csv = lt.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Prophet forecast (CSV)", csv, "prophet_forecast.csv", "text/csv")

            fig = plt.figure(figsize=(12, 4))
            x = pd.to_datetime(lt["Date"])
            plt.plot(x, lt["Expected_Load_MW"], label="Expected")
            plt.fill_between(x, lt["Min_Expected_Load_MW"], lt["Max_Expected_Load_MW"], alpha=0.2, label="Expected range")
            plt.title("Long-term Forecast (Prophet)")
            plt.xlabel("Date")
            plt.ylabel("Load (MW)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)

# -----------------------------
# Optimization (Phase 6)
# -----------------------------
with tab_opt:
    st.subheader("Optimization (Phase 6): Load Shifting + Cost Reduction")

    # Choose baseline for optimization
    actual = get_actual_data_if_available(start_date, end_date)

    if not actual.empty:
        opt_base = actual.copy()
        opt_base["Date"] = pd.to_datetime(opt_base["Date"]).dt.date
        opt_base = opt_base.rename(columns={"Grid_Load_MWh": "Expected_Load_MW"})
        opt_base = opt_base[["Date", "Expected_Load_MW"] + [c for c in weather_cols if c in opt_base.columns]]
        st.info("Using ACTUAL load as baseline for optimization (no forecast needed for this range).")
    else:
        opt_base = forecast_long_term_prophet(start_date, end_date)
        st.info("Using Prophet forecast as baseline for optimization (stable for long horizons).")

    if opt_base.empty:
        st.warning("No baseline profile available to optimize for this selection.")
    else:
        st.write(f"Optimization horizon: {start_date} → {end_date}  (days: {len(opt_base)})")

        # Guardrail message
        if len(opt_base) < 7:
            st.warning("Optimization works best with 7+ days (load shifting needs multiple days). "
                       "For 1 day, savings are not meaningful.")

        opt_results, summary = run_optimization(
            opt_base,
            flexible_fraction=flexible_fraction,
            normal_price=normal_price,
            peak_price=peak_price
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline total cost (EUR)", f"{summary['baseline_total_cost_EUR']:,.2f}")
        c2.metric("Optimized total cost (EUR)", f"{summary['optimized_total_cost_EUR']:,.2f}")
        c3.metric("Absolute savings (EUR)", f"{summary['absolute_savings_EUR']:,.2f}")
        c4.metric("Relative savings (%)", f"{summary['relative_savings_percent']:.2f}%")

        st.write("### Baseline vs Optimized Load Profile")
        fig = plt.figure(figsize=(12, 4))
        if "Energy_MWh" in opt_results.columns:
            plt.plot(pd.to_datetime(opt_results["Date"]), opt_results["Energy_MWh"] / 24.0, label="Baseline_Load_MW")
        plt.plot(pd.to_datetime(opt_results["Date"]), opt_results["Optimized_Load_MW"], label="Optimized_Load_MW")
        plt.title("Load Shifting Effect (Baseline vs Optimized)")
        plt.xlabel("Date")
        plt.ylabel("Load (MW)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        st.pyplot(fig)

        st.write("### Optimization Results (Daily)")
        show_cols = [
            "Date", "Expected_Load_MW", "Energy_MWh", "IsPeakDay", "Price_EUR_per_MWh",
            "Baseline_Cost_EUR", "Optimized_Energy_MWh", "Optimized_Load_MW", "Optimized_Cost_EUR"
        ]
        show_cols = [c for c in show_cols if c in opt_results.columns]
        st.dataframe(opt_results[show_cols].style.format({
            "Expected_Load_MW": "{:,.0f}",
            "Energy_MWh": "{:,.0f}",
            "Baseline_Cost_EUR": "{:,.0f}",
            "Optimized_Energy_MWh": "{:,.0f}",
            "Optimized_Load_MW": "{:,.0f}",
            "Optimized_Cost_EUR": "{:,.0f}",
        }), use_container_width=True)

        # Download
        csv = opt_results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download optimization results (CSV)", csv, "optimization_results.csv", "text/csv")

        # Reduction / Increase days
        if "Optimized_Energy_MWh" in opt_results.columns and "Energy_MWh" in opt_results.columns:
            reduction_days = opt_results[opt_results["Optimized_Energy_MWh"] < opt_results["Energy_MWh"]][
                ["Date", "Energy_MWh", "Optimized_Energy_MWh", "Price_EUR_per_MWh"]
            ]
            increase_days = opt_results[opt_results["Optimized_Energy_MWh"] > opt_results["Energy_MWh"]][
                ["Date", "Energy_MWh", "Optimized_Energy_MWh", "Price_EUR_per_MWh"]
            ]

            st.write("### Days recommended for reduction (expensive days)")
            st.dataframe(reduction_days.style.format({
                "Energy_MWh": "{:,.0f}",
                "Optimized_Energy_MWh": "{:,.0f}",
                "Price_EUR_per_MWh": "{:,.0f}",
            }), use_container_width=True)

            st.write("### Days recommended for increase (cheap days)")
            st.dataframe(increase_days.style.format({
                "Energy_MWh": "{:,.0f}",
                "Optimized_Energy_MWh": "{:,.0f}",
                "Price_EUR_per_MWh": "{:,.0f}",
            }), use_container_width=True)

# -----------------------------
# Model Comparison + Explainability
# -----------------------------
with tab_models:
    st.subheader("Model Comparison (Phases 4 & 5)")

    results = pd.DataFrame({
        "Model": [
            "Random Forest Regressor",
            "LightGBM",
            "LSTM",
            "Multiple Linear Regression",
            "7-day Moving Average",
            "Naive (Lag_1)",
            "Prophet",
            "SARIMAX"
        ],
        "MAE": [
            1.033977e+03,
            7.518376e+03,
            3.749348e+04,
            4.744135e+04,
            9.940179e+04,
            8.772384e+04,
            1.433177e+05,
            3.588462e+06
        ],
        "RMSE": [
            6.062954e+03,
            1.235674e+04,
            5.675075e+04,
            6.648786e+04,
            1.144000e+05,
            1.244687e+05,
            1.694525e+05,
            4.109725e+06
        ]
    }).sort_values("MAE")

    st.dataframe(results.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}"}), use_container_width=True)

    st.write("### Error Comparison (Lower is better)")
    fig = plt.figure(figsize=(12, 4))
    x = np.arange(len(results))
    width = 0.35
    plt.bar(x - width/2, results["MAE"], width, label="MAE")
    plt.bar(x + width/2, results["RMSE"], width, label="RMSE")
    plt.xticks(x, results["Model"], rotation=25, ha="right")
    plt.title("Model Comparison")
    plt.ylabel("Error")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    st.pyplot(fig)

    st.write("### Random Forest Explainability (Feature Importance)")
    if hasattr(rf_model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)

        st.dataframe(fi.style.format({"Importance": "{:.4f}"}), use_container_width=True)

        fig = plt.figure(figsize=(10, 4))
        plt.bar(fi["Feature"], fi["Importance"])
        plt.xticks(rotation=35, ha="right")
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.ylabel("Importance")
        plt.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("This Random Forest model does not expose feature_importances_.")
