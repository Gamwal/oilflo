import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit


def load_and_prepare_data(file):
    """Load and prepare the data."""
    data = pd.read_csv(file)
    return data

def filter_well_data(data, well_column, selected_well):
    """Filter data for the selected well."""
    return data[data[well_column] == selected_well].copy()

def create_lag_features(data, target_col, lags=30):
    """Generate lag features for time series data."""
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data

def generate_forecast_dates(last_date, days_to_forecast):
    """Generate forecast dates."""
    return pd.date_range(start=last_date, periods=days_to_forecast + 1, freq='D')[1:]

def plot_forecast(data, forecast_data, datetime_column, original_label="Original Rates", forecast_label="Forecasted Rates"):
    """Create and display a plotly figure for forecast data."""
    combined_df = pd.concat([
        data[[datetime_column, 'production_rate_MMSCFD']].rename(columns={'production_rate_MMSCFD': 'Rate'}),
        forecast_data.rename(columns={'Forecasted_gas_rate_MMSCFD': 'Rate'})
    ], keys=['Original', 'Forecast']).reset_index(level=0).rename(columns={'level_0': 'Type'})

    fig = go.Figure()

    # Original data
    fig.add_trace(go.Scatter(
        x=combined_df[combined_df['Type'] == 'Original'][datetime_column],
        y=combined_df[combined_df['Type'] == 'Original']['Rate'],
        mode='lines',
        name=original_label,
        line=dict(color='blue')
    ))

    # Forecast data
    fig.add_trace(go.Scatter(
        x=combined_df[combined_df['Type'] == 'Forecast'][datetime_column],
        y=combined_df[combined_df['Type'] == 'Forecast']['Rate'],
        mode='lines',
        name=forecast_label,
        line=dict(color='orange', dash='dot')
    ))

    fig.update_layout(
        title="Gas Rate Forecast",
        xaxis_title="Date",
        yaxis_title="Production Rate (MMSCFD)",
        legend_title="Rate Type",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def calculate_dca_parameters(time, rate):
    """Calculates decline curve analysis parameters (simplified)."""
    try:
        # Define a simple exponential decline function
        def exponential_decline(time, initial_rate, decline_rate):
            return initial_rate * np.exp(-decline_rate * time)

        # Fit the exponential decline function to the data
        popt, _ = curve_fit(exponential_decline, time, rate)
        initial_rate, decline_rate = popt

        return initial_rate, decline_rate
    except:
        return None, None
    
@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')