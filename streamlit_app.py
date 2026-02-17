import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="ROCE Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_main_backtest_results():
    """Load results from main ROCE increasing backtest"""
    results = {}
    
    # Check if summary file exists
    summary_file = 'backtest_summary.csv'
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        
        for _, row in summary_df.iterrows():
            years = int(row['Years_ROCE_Increasing'])
            key = f"ROCE_Increasing_{years}years"
            
            # Load portfolio history
            portfolio_file = f'portfolio_history_{years}years.csv'
            transactions_file = f'transactions_{years}years.csv'
            
            portfolio_history = None
            transactions = None
            
            if os.path.exists(portfolio_file):
                portfolio_history = pd.read_csv(portfolio_file)
                portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
            
            if os.path.exists(transactions_file):
                transactions = pd.read_csv(transactions_file)
                if 'date' in transactions.columns:
                    transactions['date'] = pd.to_datetime(transactions['date'])
            
            results[key] = {
                'strategy_type': 'ROCE Increasing',
                'strategy_name': f'{years} Year(s) ROCE Increasing',
                'years': years,
                'initial_capital': row['Initial_Capital'],
                'final_value': row['Final_Value'],
                'total_return_pct': row['Total_Return_Pct'],
                'cagr_pct': row['CAGR_Pct'],
                'years_elapsed': row['Years_Elapsed'],
                'num_transactions': row['Num_Transactions'],
                'num_buys': row['Num_Buys'],
                'num_sells': row['Num_Sells'],
                'portfolio_history': portfolio_history,
                'transactions': transactions
            }
    
    return results

@st.cache_data
def load_simple_backtest_results():
    """Load results from simple ROCE threshold backtest"""
    results = {}
    results_dir = 'simple_backtest_results'
    
    summary_file = os.path.join(results_dir, 'backtest_summary.csv')
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        
        for _, row in summary_df.iterrows():
            roce_threshold = int(row['ROCE_Threshold'])
            holding_period = int(row['Holding_Period_Years'])
            key = f"ROCE>{roce_threshold}_Hold{holding_period}y"
            
            # Load portfolio history
            portfolio_file = os.path.join(results_dir, f'portfolio_history_{key}.csv')
            transactions_file = os.path.join(results_dir, f'transactions_{key}.csv')
            
            portfolio_history = None
            transactions = None
            
            if os.path.exists(portfolio_file):
                portfolio_history = pd.read_csv(portfolio_file)
                portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
            
            if os.path.exists(transactions_file):
                transactions = pd.read_csv(transactions_file)
                if 'date' in transactions.columns:
                    transactions['date'] = pd.to_datetime(transactions['date'])
            
            results[key] = {
                'strategy_type': 'ROCE Threshold',
                'strategy_name': f'ROCE > {roce_threshold}%, Hold {holding_period} Year(s)',
                'roce_threshold': roce_threshold,
                'holding_period': holding_period,
                'initial_capital': row['Initial_Capital'],
                'final_value': row['Final_Value'],
                'total_return_pct': row['Total_Return_Pct'],
                'cagr_pct': row['CAGR_Pct'],
                'years_elapsed': row['Years_Elapsed'],
                'num_transactions': row['Num_Transactions'],
                'num_buys': row['Num_Buys'],
                'num_sells': row['Num_Sells'],
                'portfolio_history': portfolio_history,
                'transactions': transactions
            }
    
    return results

def create_portfolio_chart(portfolio_history, strategy_name, initial_capital=None):
    """Create interactive portfolio value chart"""
    if portfolio_history is None or len(portfolio_history) == 0:
        return None
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>Portfolio Value:</b> ₹%{y:,.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Initial capital line
    if initial_capital is None:
        # Try to get from first portfolio value if available
        if len(portfolio_history) > 0 and 'portfolio_value' in portfolio_history.columns:
            initial_capital = portfolio_history['portfolio_value'].iloc[0]
    
    if initial_capital is not None:
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="#A23B72",
            annotation_text="Initial Capital",
            annotation_position="right"
        )
    
    fig.update_layout(
        title=f'Portfolio Value Over Time - {strategy_name}',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        yaxis=dict(tickformat=',.0f')
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📈 ROCE Backtest Dashboard</div>', unsafe_allow_html=True)
    
    # Load results
    with st.spinner("Loading backtest results..."):
        main_results = load_main_backtest_results()
        simple_results = load_simple_backtest_results()
        all_results = {**main_results, **simple_results}
    
    if len(all_results) == 0:
        st.error("No backtest results found. Please run the backtests first.")
        st.info("Run `python3 backtest.py` and `python3 simple_backtest.py` to generate results.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Strategy type filter
    strategy_types = list(set([r['strategy_type'] for r in all_results.values()]))
    selected_strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        options=['All'] + strategy_types,
        index=0
    )
    
    # Filter results by strategy type
    filtered_results = all_results
    if selected_strategy_type != 'All':
        filtered_results = {k: v for k, v in all_results.items() 
                           if v['strategy_type'] == selected_strategy_type}
    
    # Sort options
    sort_options = {
        'Best CAGR': 'cagr_pct',
        'Best Total Return': 'total_return_pct',
        'Highest Final Value': 'final_value',
        'Most Transactions': 'num_transactions'
    }
    
    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=list(sort_options.keys()),
        index=0
    )
    
    # Sort results
    sorted_results = sorted(
        filtered_results.items(),
        key=lambda x: x[1][sort_options[sort_by]],
        reverse=True
    )
    
    # Strategy selection
    strategy_options = {k: v['strategy_name'] for k, v in sorted_results}
    selected_strategy_key = st.sidebar.selectbox(
        "Select Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=0
    )
    
    selected_strategy = filtered_results[selected_strategy_key]
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 {selected_strategy['strategy_name']}")
    
    with col2:
        st.markdown("### Strategy Details")
        if selected_strategy['strategy_type'] == 'ROCE Increasing':
            st.info(f"**Type:** ROCE Increasing Strategy\n\n**Years:** {selected_strategy['years']} year(s)")
        else:
            st.info(f"**Type:** ROCE Threshold Strategy\n\n**ROCE Threshold:** >{selected_strategy['roce_threshold']}%\n\n**Holding Period:** {selected_strategy['holding_period']} year(s)")
    
    # Performance Metrics
    st.subheader("📈 Performance Metrics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            "CAGR",
            f"{selected_strategy['cagr_pct']:.2f}%",
            delta=f"{selected_strategy['cagr_pct']:.2f}%"
        )
    
    with metric_cols[1]:
        st.metric(
            "Total Return",
            f"{selected_strategy['total_return_pct']:.2f}%",
            delta=f"{selected_strategy['total_return_pct']:.2f}%"
        )
    
    with metric_cols[2]:
        initial_cap = selected_strategy['initial_capital']
        final_val = selected_strategy['final_value']
        st.metric(
            "Final Value",
            f"₹{final_val:,.0f}",
            delta=f"₹{final_val - initial_cap:,.0f}"
        )
    
    with metric_cols[3]:
        st.metric(
            "Years Elapsed",
            f"{selected_strategy['years_elapsed']:.2f}",
        )
    
    # Additional metrics
    st.subheader("📋 Additional Metrics")
    
    metric_cols2 = st.columns(4)
    
    with metric_cols2[0]:
        st.metric("Initial Capital", f"₹{selected_strategy['initial_capital']:,.0f}")
    
    with metric_cols2[1]:
        st.metric("Total Transactions", f"{selected_strategy['num_transactions']}")
    
    with metric_cols2[2]:
        st.metric("Buy Signals", f"{selected_strategy['num_buys']}")
    
    with metric_cols2[3]:
        st.metric("Sell Signals", f"{selected_strategy['num_sells']}")
    
    # Portfolio Chart
    st.subheader("📊 Portfolio Value Chart")
    
    if selected_strategy['portfolio_history'] is not None and len(selected_strategy['portfolio_history']) > 0:
        fig = create_portfolio_chart(
            selected_strategy['portfolio_history'],
            selected_strategy['strategy_name'],
            initial_capital=selected_strategy['initial_capital']
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Portfolio history data not available for this strategy.")
    
    # Download Section
    st.subheader("💾 Download Data")
    
    download_cols = st.columns(2)
    
    with download_cols[0]:
        if selected_strategy['portfolio_history'] is not None and len(selected_strategy['portfolio_history']) > 0:
            portfolio_csv = selected_strategy['portfolio_history'].to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio History (CSV)",
                data=portfolio_csv,
                file_name=f"portfolio_history_{selected_strategy_key}.csv",
                mime="text/csv"
            )
        else:
            st.info("Portfolio history not available")
    
    with download_cols[1]:
        if selected_strategy['transactions'] is not None and len(selected_strategy['transactions']) > 0:
            transactions_csv = selected_strategy['transactions'].to_csv(index=False)
            st.download_button(
                label="📥 Download Transactions (CSV)",
                data=transactions_csv,
                file_name=f"transactions_{selected_strategy_key}.csv",
                mime="text/csv"
            )
        else:
            st.info("Transactions data not available")
    
    # Strategy Comparison Table
    st.subheader("📊 All Strategies Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for key, result in sorted_results:
        comparison_data.append({
            'Strategy': result['strategy_name'],
            'Type': result['strategy_type'],
            'CAGR (%)': result['cagr_pct'],
            'Total Return (%)': result['total_return_pct'],
            'Final Value (₹)': result['final_value'],
            'Initial Capital (₹)': result['initial_capital'],
            'Years Elapsed': result['years_elapsed'],
            'Transactions': result['num_transactions'],
            'Buys': result['num_buys'],
            'Sells': result['num_sells']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Highlight selected strategy
    def highlight_selected(row):
        if row['Strategy'] == selected_strategy['strategy_name']:
            return ['background-color: #ffd700'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        comparison_df.style.apply(highlight_selected, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download comparison table
    comparison_csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Table (CSV)",
        data=comparison_csv,
        file_name="all_strategies_comparison.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>ROCE Backtest Dashboard | Generated from backtest results</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

