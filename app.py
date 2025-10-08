import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta

# Try to import Prophet (for advanced forecasting)
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

# --------------------------
# APP TITLE
# --------------------------
st.set_page_config(page_title="AI Business Forecasting Suite", layout="wide")
st.title("📊 AI Business Forecasting Suite")
st.write("Upload your CSV or Excel file with date and sales columns to generate forecasts automatically and measure model accuracy.")

# --------------------------
# FILE UPLOAD SECTION
# --------------------------
uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Detect file type and load
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("✅ File uploaded successfully!")
    st.dataframe(data.head())

    # --------------------------
    # AUTO-DETECT DATE AND SALES COLUMNS
    # --------------------------
    date_col = None
    sales_col = None
    for col in data.columns:
        if "date" in col.lower():
            date_col = col
        if "sale" in col.lower() or "revenue" in col.lower() or "amount" in col.lower():
            sales_col = col

    if not date_col or not sales_col:
        st.error("❌ Could not automatically detect 'date' or 'sales' column. Please rename them properly.")
        st.stop()

    # Clean & prepare data
    data = data[[date_col, sales_col]].copy()
    data.columns = ["date", "sales"]
    data.dropna(inplace=True)

    # Convert to datetime
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data.sort_values("date", inplace=True)

    if len(data) == 0:
        st.error("❌ No valid data rows found after cleaning. Please check your file.")
        st.stop()

    # --------------------------
    # DISPLAY CLEANED DATA
    # --------------------------
    st.subheader("🧹 Cleaned Data Preview")
    st.dataframe(data.tail())

    # Plot existing data
    st.subheader("📈 Current Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(data["date"], data["sales"], marker="o", linestyle="-")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Over Time")
    st.pyplot(fig)

    # --------------------------
    # FORECASTING SECTION
    # --------------------------
    st.subheader("🔮 Forecasting")
    forecast_periods = st.slider("Select number of future days to forecast", 7, 90, 30)

    if st.button("🚀 Generate Forecast"):
        with st.spinner("Generating forecast and evaluating model..."):
            if prophet_available:
                # Prophet-based Forecast
                df = data.rename(columns={"date": "ds", "sales": "y"})
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=forecast_periods)
                forecast = model.predict(future)

                st.success("✅ Forecast generated using Prophet!")

                # Evaluate on known data only
                merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
                mae = mean_absolute_error(merged["y"], merged["yhat"])
                r2 = r2_score(merged["y"], merged["yhat"])

                st.metric("📏 Mean Absolute Error (MAE)", f"{mae:.2f}")
                st.metric("🎯 R² Score", f"{r2:.3f}")

                # Display forecasted data
                st.subheader("📈 Forecast Results")
                st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_periods))

                # Download button
                csv_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_periods).to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name="forecast_results.csv",
                    mime="text/csv"
                )

                # Plot forecast
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

            else:
                # Linear Regression fallback
                st.warning("⚠ Prophet not installed. Using Linear Regression model instead.")
                data["day_number"] = range(len(data))
                X = data[["day_number"]]
                y = data["sales"]

                model = LinearRegression()
                model.fit(X, y)

                # Evaluate on known data
                preds = model.predict(X)
                mae = mean_absolute_error(y, preds)
                r2 = r2_score(y, preds)

                st.metric("📏 Mean Absolute Error (MAE)", f"{mae:.2f}")
                st.metric("🎯 R² Score", f"{r2:.3f}")

                # Forecast next N days
                future_days = list(range(len(data), len(data) + forecast_periods))
                future_dates = [data["date"].max() + timedelta(days=i + 1) for i in range(forecast_periods)]
                predictions = model.predict(pd.DataFrame({"day_number": future_days}))

                forecast_df = pd.DataFrame({"date": future_dates, "forecasted_sales": predictions})
                st.success("✅ Forecast generated using Linear Regression!")

                st.subheader("📈 Forecast Results")
                st.dataframe(forecast_df)

                # Download button
                csv_data = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name="forecast_results.csv",
                    mime="text/csv"
                )

                # Plot results
                fig, ax = plt.subplots()
                ax.plot(data["date"], data["sales"], label="Actual", marker="o")
                ax.plot(forecast_df["date"], forecast_df["forecasted_sales"], label="Forecast", marker="x")
                ax.legend()
                ax.set_title("Sales Forecast")
                st.pyplot(fig)
else:
    st.info("👆 Please upload a CSV or Excel file to begin.")
