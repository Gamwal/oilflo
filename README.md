## OilFlo: Reservoir Production Forecasting Tool

**OilFlo** is a web-based application designed to streamline reservoir production forecasting. Built with Python, Streamlit, and Plotly, it provides users with intuitive tools for analyzing data, modeling production trends, and generating long-term forecasts using Arps decline models.

**Benefits:**

- **Optimize reservoir management:** Make informed decisions based on accurate production forecasts.
- **For everyone in oil & gas:** Useful for reservoir engineers, data scientists, and decision-makers.
- **Streamlined workflow:** Easy data upload, interactive visualizations, and clear forecasting.

**Features:**

1. **Exploratory Data Analysis (EDA):**

   - Upload production data in CSV format.
   - Select key parameters like wells and time columns.
   - View and filter data interactively.
   - Generate crossplots to visualize relationships.

2. **Production Trend Modeling:**

   - Supports Exponential, Harmonic, and Hyperbolic decline models.
   - Automatic model fitting with optimized parameters.
   - Visualize actual vs. predicted production rates for validation.

3. **Forecasting:**

   - Forecast production using Arps decline models.
   - Customize input parameters:
     - Initial production rate (Qi)
     - Decline rate (Di)
     - Decline exponent (b) for Hyperbolic models
     - Economic limit rate (Qel)
   - Calculate cumulative production and daily forecasted rates.
   - Download forecast results for further analysis.

4. **Interactive Visualizations:**

   - Dynamic plots of production rates and forecasted trends.
   - High-quality, interactive graphs with Plotly.

5. **User-Friendly Interface:**
   - Sleek, modern, and responsive experience built with Streamlit.
   - Logical flow from data upload to forecasting.

**Installation:**

1. Clone the repository:

```bash
git clone https://github.com/Gamwal/oilflo.git
cd oilflo
```

2. Set up a Python virtual environment:

- Using Pip:

```bash
python3 -m venv [virtual_environment]
source [virtual_environment]/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch the application:

```bash
streamlit run OilFlo\ v2\ Beta.py
```

**How to Use:**

1. Upload a CSV file containing production data (well names, dates, production volumes, operational hours)
2. Select well identifiers and date-time values.
3. Explore data and identify trends using interactive tools.
4. Fit a decline model, define the training period, and visualize the model fit.
5. Input forecast parameters, select the timeframe, and view results. Download forecasts.
6. Compare historical data with forecasted trends using Plotly visualizations.

**Key Technologies:**

- Python (core programming)
- Streamlit (user interface)
- Plotly (data visualizations)
- Pandas & NumPy (data manipulation)
- SciPy (optimization and curve fitting)

**Acknowledgements:**

OilFlo utilizes robust libraries and engineering principles. Special thanks to contributors and open-source communities.

**Future Enhancements:**

- Machine learning models for production forecasting.
- Multi-well forecasting support.
- Database and real-time data stream integration.

**Start forecasting with confidence! **
