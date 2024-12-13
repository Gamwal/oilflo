import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
import plotly.graph_objects as go
from arps_equations import *
from utils import *

# Display header
st.image(image="OilFlo.png")
st.subheader("Your goto Reservoir Production Forecasting Tool")

# File upload
uploaded_file = st.file_uploader(label="Upload your production data here", type=['csv'])
if not uploaded_file:
    st.stop()

data = load_and_prepare_data(uploaded_file)

# Column selection
col6, col7 = st.columns(2)
with col6:
    wellnames_column = st.selectbox("Wellname/UniqueID", data.columns, index=None)
with col7:
    datetime_column = st.selectbox("Time/Date", data.columns, index=None)

if not wellnames_column or not datetime_column:
    st.stop()

# Convert datetime column
data[datetime_column] = pd.to_datetime(data[datetime_column])

# Show uploaded data
with st.expander("View Table"):
    st.dataframe(data)

#%%
# Well selection and filtering
st.subheader("Exploratory Data Analysis")
col3, col4, col5 = st.columns(3)
with col3:
    picked_well = st.selectbox(
        "Which well would you like to view?",
        data[wellnames_column].unique(),
        placeholder="Select well to view...",
        index=42
    )

if not picked_well:
    st.stop()

# Filter data
data_subset = filter_well_data(data, wellnames_column, picked_well)
data_subset = data_subset[(data_subset['ALLOC_GAS_VOL_MMSCF'] > 0) & (data_subset['HRS_ON'] >= 23)]
data_subset['production_rate_MMSCFD'] = data_subset['ALLOC_GAS_VOL_MMSCF']/(data_subset['HRS_ON']/24)
data_subset['time'] = (data_subset[datetime_column] - data_subset[datetime_column].min()).dt.days

# Remove irrelevant columns
with col4:
    x_column = st.selectbox('X-axis', data_subset.columns, index=None)
with col5:
    y_column = st.selectbox('Y-axis', data_subset.columns, index=None)

# View filtered data
with st.expander("View Well Table"):
    st.dataframe(data_subset)

# Crossplots
if not x_column or not y_column:
    st.stop()

with st.expander("View Crossplots"):
    st.scatter_chart(data_subset, x=x_column, y=y_column, color=None)

#%%
# Model Fitting
st.subheader(f"Forecast for Well: {picked_well}")

# Select decline model
decline_model = st.selectbox(
    "Select Decline Model",
    ["Exponential", "Harmonic", "Hyperbolic"]
)

start_time, end_time = st.select_slider(
    "Select your preferred production time range to fit",
    options=sorted(data_subset[datetime_column]),  # Ensure sorted options
    value=[data_subset[datetime_column].min(), data_subset[datetime_column].max()]
)

selected_data = data_subset[(data_subset[datetime_column] >= start_time) & (data_subset[datetime_column] <= end_time)]

# Cut off the last 40% of data for modeling
cutoff_index = int(len(selected_data) * 0.8)
training_data = selected_data.iloc[:cutoff_index]
testing_data = selected_data.iloc[cutoff_index:]

# Time and production rate for training
t_train = training_data['time']
q_train = training_data['production_rate_MMSCFD']

# Time and production rate for testing
t_test = testing_data['time']
q_test = testing_data['production_rate_MMSCFD']

qi = q_train.max()

# Fit model to training data
try:
    if decline_model == "Exponential":
        bounds = ([qi - 0.01, 0.0001], [qi, 0.01])
        popt, _ = curve_fit(ArpsRateExp, t_train, q_train , bounds=bounds, method='trf')
        training_data['Pred_Rate'] = ArpsRateExp(t_train, popt[0], popt[1])
        testing_data['Pred_Rate'] = ArpsRateExp(t_test, popt[0], popt[1])
        st.write(f"Estimated constants are Qi: {popt[0]:.2f} | Di: {popt[1]*365:.2f} | b: {0}")
    elif decline_model == "Harmonic":
        bounds = ([qi - 0.01, 0.0001], [qi, 0.01])
        popt, _ = curve_fit(ArpsRateHar, t_train, q_train, bounds=bounds, method='trf')
        training_data['Pred_Rate'] = ArpsRateHar(t_train, popt[0], popt[1])
        testing_data['Pred_Rate'] = ArpsRateHar(t_test, popt[0], popt[1])
        st.write(f"Estimated constants are Qi: {popt[0]:.2f} | Di: {popt[1]*365:.2f} | b: {1}")
    elif decline_model == "Hyperbolic":
        bounds = ([qi - 0.01, 0.0001, 0.1], [qi, 0.01, 0.9])
        popt, _ = curve_fit(ArpsRateHyp, t_train, q_train, bounds=bounds, method='trf')
        training_data['Pred_Rate'] = ArpsRateHyp(t_train, popt[0], popt[1], popt[2])
        testing_data['Pred_Rate'] = ArpsRateHyp(t_test, popt[0], popt[1], popt[2])
        st.write(f"Estimated constants are Qi: {popt[0]:.2f} | Di: {popt[1]*365:.2f} | b: {popt[2]:.2f}")
except ValueError:
    st.write("Adjust your time selection")
    st.stop()

# Combine training and testing data for visualization
all_data = pd.concat([training_data, testing_data])

#st.dataframe(all_data)

# Plot actual vs predicted
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=data_subset[datetime_column],
    y=data_subset['production_rate_MMSCFD'],
    mode='lines',
    name='Actual Rates',
    line=dict(color='blue')
))

# Add predicted data
fig.add_trace(go.Scatter(
    x=all_data[datetime_column],
    y=all_data['Pred_Rate'],
    mode='lines',
    name='Predicted Rates',
    line=dict(color='red', dash='dot')
))

# Customize layout
fig.update_layout(
    title=f"Model Validation for Well: {picked_well} ({decline_model} Model)",
    xaxis_title="Date",
    yaxis_title="Gas Rate (MMSCFD)",
    legend_title="Rate Type",
    template="plotly_white",
    height=500
)

# Display plot
st.plotly_chart(fig, use_container_width=True)


#%%
#Forecasting
Qi_forecast = data_subset[y_column].iloc[-50:].mean()

# Input Arps parameters
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    qi = st.number_input("Initial Production Rate (Qi) [MMSCFD]", value=Qi_forecast, min_value=0.0, step=0.1)
with col2:
    di = st.number_input("Initial Decline Rate (Di) [percent/year]", value=popt[1]*365, min_value=0.0, step=0.01)
with col3:
    b = st.number_input("Decline Exponent (b, for Hyperbolic)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
with col4:
    qel = st.number_input("Economic Limit Rate (Qel) [MMSCFD]", value=0.10, min_value=0.0, step=0.1)
with col5:
    unit = st.selectbox("What are the units of your volumes",
                        ("MMSCF", "BBLS"))

# Forecast duration
years_to_forecast = st.number_input("Number of years to forecast?", value=10)


if st.button("Run Arps Forecast"):
    # Generate forecast dates
    days_to_forecast = 365 * years_to_forecast
    forecast_dates = pd.date_range(
        start=data_subset[datetime_column].iloc[-1],
        periods=days_to_forecast + 1,
        freq="D"
    )[1:]  # Skip the first date

    # Calculate production rates
    time_years = np.arange(1, days_to_forecast + 1) / 365  # Time in years
    if decline_model == "Exponential":
        production_rates = ArpsRateExp(time_years, qi, di)
    elif decline_model == "Harmonic":
        production_rates = ArpsRateHar(time_years, qi, di)
    elif decline_model == "Hyperbolic":
        production_rates = ArpsRateHyp(time_years, qi, di, b)

    # Cumulative production calculation
    d = NominalDecline(di, time_years.max(), b)
    cumulative_production = ArpsCumProd(qi, qel, d, b)

    # st.write(cumulative_production)

    # Create forecast DataFrame
    arps_forecast_df = pd.DataFrame({
        datetime_column: forecast_dates,
        'Forecasted_gas_rate_MMSCFD': production_rates
    })

    arps_forecast_df = arps_forecast_df[arps_forecast_df['Forecasted_gas_rate_MMSCFD'] >= qel]

    # Display forecast table
    st.write(f"Daily Forecasted Production Rates (Next {years_to_forecast} Years):")
    with st.expander("View Forecast Table"):
        st.dataframe(arps_forecast_df)

    # Display cumulative production
    st.write(f"Estimated Cumulative Production: {arps_forecast_df['Forecasted_gas_rate_MMSCFD'].sum():.2f} {unit}")

    # Combine original and forecast data
    combined_df = pd.concat([
        data_subset[[datetime_column, 'production_rate_MMSCFD']].rename(columns={'production_rate_MMSCFD': 'Rate'}),
        arps_forecast_df.rename(columns={'Forecasted_gas_rate_MMSCFD': 'Rate'})
    ], keys=['Original', 'Forecast']).reset_index(level=0).rename(columns={'level_0': 'Type'})

    # Create a plotly figure
    fig = go.Figure()

    # Add original data
    fig.add_trace(go.Scatter(
        x=combined_df[combined_df['Type'] == 'Original'][datetime_column],
        y=combined_df[combined_df['Type'] == 'Original']['Rate'],
        mode='lines',
        name='Original Rates',
        line=dict(color='blue')
    ))

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=combined_df[combined_df['Type'] == 'Forecast'][datetime_column],
        y=combined_df[combined_df['Type'] == 'Forecast']['Rate'],
        mode='lines',
        name='Forecasted Rates',
        line=dict(color='green', dash='dot')
    ))

    # Customize layout
    fig.update_layout(
        title=f"Production Rate Forecast for Well: {picked_well} ({decline_model} Model)",
        xaxis_title="Date",
        yaxis_title="Production Rate (MMSCFD)",
        legend_title="Rate Type",
        template="plotly_white",
        height=500
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)
# %%
# Download the forecast table   
    forecast_table = convert_df(arps_forecast_df)
    st.download_button(label='Download the forecst table here', data=forecast_table, mime='text/csv', file_name='prediction_table.csv')