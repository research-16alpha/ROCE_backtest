# ROCE Backtest Dashboard - Streamlit App

## Overview
This Streamlit dashboard allows you to visualize and explore backtest results from both ROCE strategies:
1. **ROCE Increasing Strategy**: Buys stocks when ROCE increases for 1, 2, 3, or 4 consecutive years
2. **ROCE Threshold Strategy**: Buys stocks with ROCE above thresholds (10%, 15%, 20%, 25%, 30%) and holds for 1-5 years

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy plotly
```

## Running the Dashboard

1. Make sure you have run the backtests first:
```bash
python3 backtest.py
python3 simple_backtest.py
```

2. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser automatically (usually at `http://localhost:8501`)

## Features

### 🔍 Filters
- **Strategy Type**: Filter by "ROCE Increasing" or "ROCE Threshold" strategies
- **Sort By**: Sort strategies by:
  - Best CAGR
  - Best Total Return
  - Highest Final Value
  - Most Transactions

### 📊 Performance Metrics
For each selected strategy, the dashboard displays:
- **CAGR**: Compound Annual Growth Rate
- **Total Return**: Overall return percentage
- **Final Value**: Portfolio value at the end
- **Years Elapsed**: Duration of the backtest
- **Initial Capital**: Starting capital
- **Total Transactions**: Number of buy/sell transactions
- **Buy/Sell Signals**: Count of buy and sell signals

### 📈 Portfolio Chart
Interactive Plotly chart showing:
- Portfolio value over time
- Initial capital reference line
- Hover tooltips with exact values

### 💾 Download Options
- **Portfolio History CSV**: Daily portfolio value data
- **Transactions CSV**: All buy/sell transactions
- **Comparison Table CSV**: Summary of all strategies

### 📊 Strategy Comparison
Interactive table showing all strategies with:
- Performance metrics
- Highlighted selected strategy
- Sortable columns

## Usage Tips

1. **Find Best Strategy**: Use the "Sort By" filter to find the best performing strategy by CAGR or Total Return
2. **Compare Strategies**: Use the comparison table to see all strategies side-by-side
3. **Download Data**: Click download buttons to export data for further analysis
4. **Interactive Charts**: Hover over the portfolio chart to see exact values at any date

## Troubleshooting

- **No results found**: Make sure you've run both backtest scripts first
- **Missing data**: Check that CSV files exist in the current directory and `simple_backtest_results/` folder
- **Port not available**: If port 8501 is busy, Streamlit will try the next available port

