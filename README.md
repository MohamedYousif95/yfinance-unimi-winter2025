# yfinance-unimi-winter2025

This is an analytical project on yahoo finance data for the "Coding for Data Science and Data Management" at UNIMI

## 📊 Project Description

An interactive financial data analysis dashboard built with Streamlit and yfinance. This application provides comprehensive tools for analyzing stock market data, including:

- **Portfolio Analysis**: Track and compare multiple stock tickers
- **Performance Metrics**: Calculate CAGR, volatility and tail risk measures (VaR/CVaR)
- **Risk Analysis**: Compute rolling beta, correlation matrices, and drawdown analysis
- **Interactive Visualizations**: Dynamic charts using Plotly for returns distribution, risk-return scatter plots, and time series analysis
- **Statistical Insights**: Daily and monthly return distributions, rolling statistics, and performance benchmarking

## 🚀 Installation

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <git@github.com:MohamedYousif95/yfinance-unimi-winter2025.git>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

The project requires the following core packages:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `plotly` - Interactive visualizations
- `matplotlib` - Additional plotting (for notebooks)
- `streamlit` - Web application framework
- `yfinance` - Yahoo Finance data fetching
- `jupyter` - Jupyter notebook support
- `ipykernel` - Python kernel for Jupyter


## 🎯 How to Run

### Running the Streamlit Web App

1. **Launch the application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the dashboard**
   - The app will open automatically in your default browser

3. **Using the dashboard**
   - Select tickers from the sidebar
   - Choose date range and data interval
   - Navigate through tabs for different analyses


## 📁 Project Structure

```
yfinance-unimi-winter2025/
├── streamlit_app.py              # Main Streamlit application (functional)
├── requirements.txt              # Project dependencies
├── README.md
├── src/
│   └── yfinance_app/
│       ├── __init__.py
│       ├── config.py             # Configuration settings
│       ├── data.py               # Data fetching functions
│       ├── transform.py          # Data transformation functions
│       ├── stats.py              # Statistical calculations
│       └── visualization.py      # Plotting functions
├── notebooks/
│   └── test_notebook.ipynb       # Jupyter notebook for analysis
```

## 🔧 Features

### Data Analysis
- Fetch historical stock data from Yahoo Finance
- Clean and preprocess financial time series
- Calculate daily and monthly returns

### Performance Metrics
- **CAGR**: Compound Annual Growth Rate
- **Volatility**: Annualized standard deviation
- **VaR/CVaR**: Value at Risk and Conditional Value at Risk
- **Maximum Drawdown**: Largest peak-to-trough decline

### Visualizations
- Normalized price charts
- Return distribution histograms
- Correlation heatmaps
- Risk-return scatter plots
- Rolling beta analysis
- Drawdown charts
- Rolling volatility plots

## 🎓 Course Information

**Course**: Coding for Data Science and Data Management
**Institution**: Università degli Studi di Milano (UNIMI)
**Academic Year**: Winter 2025


## 👥 Contributor

[Mohamed Yousif]