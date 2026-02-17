import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib for graphing
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Graphs will not be generated.")
    print("Install with: pip install matplotlib")

class ROCEBacktest:
    def __init__(self, roce_file, price_file, initial_capital=10000000, max_stocks=200, weight_per_stock=0.005, start_date='2018-01-01'):
        """
        Initialize the backtest
        
        Parameters:
        -----------
        roce_file: str
            Path to ROCE CSV file
        price_file: str
            Path to close price CSV file
        initial_capital: float
            Initial capital (default: 1,00,00,000)
        max_stocks: int
            Maximum number of stocks in portfolio (default: 200)
        weight_per_stock: float
            Weight per stock (default: 0.005 = 0.5%)
        start_date: str
            Start date for filtering data (default: '2018-01-01')
        """
        self.initial_capital = initial_capital
        self.max_stocks = max_stocks
        self.weight_per_stock = weight_per_stock
        self.roce_file = roce_file
        self.price_file = price_file
        self.start_date = start_date
        
        # Load and clean data
        self.load_and_clean_data(start_date=start_date)
        
    def load_and_clean_data(self, start_date='2018-01-01'):
        """
        Load ROCE and price data, find intersection of companies
        
        Parameters:
        -----------
        start_date: str
            Start date for filtering data (default: '2018-01-01')
        """
        print("Loading ROCE data...")
        # ROCE: Companies in first column, dates in first row
        self.roce_df = pd.read_csv(self.roce_file)
        self.roce_df.set_index('Company Name', inplace=True)
        
        # Convert date columns to datetime
        self.roce_df.columns = pd.to_datetime(self.roce_df.columns, format='%b %Y', errors='coerce')
        
        # Filter ROCE data to only include dates from start_date onwards
        start_datetime = pd.to_datetime(start_date)
        valid_roce_columns = [col for col in self.roce_df.columns 
                              if pd.notna(col) and col >= start_datetime]
        if len(valid_roce_columns) > 0:
            self.roce_df = self.roce_df[valid_roce_columns]
            print(f"  Filtered ROCE data: Using dates from {start_date} onwards ({len(valid_roce_columns)} date columns)")
        else:
            print(f"  Warning: No ROCE data found from {start_date} onwards")
        
        print("Loading close price data...")
        # Close price: Dates in first column, companies in first row
        self.price_df = pd.read_csv(self.price_file)
        self.price_df['date'] = pd.to_datetime(self.price_df['date'])
        self.price_df.set_index('date', inplace=True)
        
        # Filter price data to only include dates from start_date onwards
        start_datetime = pd.to_datetime(start_date)
        self.price_df = self.price_df[self.price_df.index >= start_datetime]
        print(f"  Filtered price data: Using dates from {start_date} onwards ({len(self.price_df)} rows)")
        
        # Clean column names (remove extra spaces)
        self.price_df.columns = self.price_df.columns.str.strip()
        
        # Find intersection of company names
        roce_companies = set(self.roce_df.index)
        price_companies = set(self.price_df.columns)
        
        # Try to match company names (case-insensitive, handle spaces)
        matched_companies = []
        roce_to_price_mapping = {}
        
        for roce_company in roce_companies:
            # Try exact match first
            if roce_company in price_companies:
                matched_companies.append(roce_company)
                roce_to_price_mapping[roce_company] = roce_company
            else:
                # Try case-insensitive match
                roce_lower = roce_company.lower().strip()
                for price_company in price_companies:
                    if price_company.lower().strip() == roce_lower:
                        matched_companies.append(roce_company)
                        roce_to_price_mapping[roce_company] = price_company
                        break
        
        print(f"Found {len(matched_companies)} matching companies out of {len(roce_companies)} ROCE companies and {len(price_companies)} price companies")
        
        # Filter data to only matched companies
        self.roce_df = self.roce_df.loc[matched_companies]
        self.price_df = self.price_df[[roce_to_price_mapping[c] for c in matched_companies]]
        self.price_df.columns = matched_companies  # Rename to match ROCE index
        
        self.companies = matched_companies
        print(f"Data cleaned. Working with {len(self.companies)} companies")
        
    def check_roce_increasing(self, company, date, years):
        """
        Check if ROCE is increasing for 'years' consecutive years before 'date'
        
        Parameters:
        -----------
        company: str
            Company name
        date: datetime
            Date to check
        years: int
            Number of years ROCE should be increasing
        
        Returns:
        --------
        bool: True if ROCE is increasing for 'years' years
        """
        if company not in self.roce_df.index:
            return False
        
        # Get ROCE values for this company
        roce_values = self.roce_df.loc[company]
        
        # Find dates before or equal to the check date
        valid_dates = [d for d in roce_values.index if pd.notna(d) and d <= date]
        if len(valid_dates) < years + 1:  # Need at least years+1 data points
            return False
        
        # Sort dates and get the last years+1 values
        valid_dates = sorted(valid_dates, reverse=True)[:years+1]
        valid_dates = sorted(valid_dates)  # Sort ascending for comparison
        
        # Get ROCE values for these dates
        roce_sequence = []
        for d in valid_dates:
            value = roce_values[d]
            if pd.isna(value) or value == '':
                return False
            try:
                roce_sequence.append(float(value))
            except (ValueError, TypeError):
                return False
        
        # Check if ROCE is strictly increasing
        for i in range(1, len(roce_sequence)):
            if roce_sequence[i] <= roce_sequence[i-1]:
                return False
        
        return True
    
    def get_closest_price(self, company, target_date, max_days=15):
        """
        Get closest available price for a company on or after target_date
        
        Parameters:
        -----------
        company: str
            Company name
        target_date: datetime
            Target date
        max_days: int
            Maximum days to look ahead (default: 15)
        
        Returns:
        --------
        tuple: (price, date) or (None, None) if not found within max_days
        """
        if company not in self.price_df.columns:
            return None, None
        
        # Get prices for this company
        prices = self.price_df[company].dropna()
        
        # Find dates on or after target_date
        future_dates = prices[prices.index >= target_date]
        
        if len(future_dates) == 0:
            return None, None
        
        # Get the first available date
        first_date = future_dates.index[0]
        
        # Check if within max_days
        days_diff = (first_date - target_date).days
        if days_diff > max_days:
            return None, None
        
        price = future_dates.iloc[0]
        if pd.isna(price) or price <= 0:
            return None, None
        
        return float(price), first_date
    
    def run_backtest(self, years_list=[1, 2, 3, 4]):
        """
        Run backtest for different year criteria
        
        Parameters:
        -----------
        years_list: list
            List of years to test (default: [1, 2, 3, 4])
        
        Returns:
        --------
        dict: Results for each year criteria
        """
        results = {}
        
        for years in years_list:
            print(f"\n{'='*60}")
            print(f"Running backtest for {years} year(s) ROCE increasing criteria")
            print(f"{'='*60}")
            
            portfolio = {}  # {company: {'entry_date': date, 'entry_price': price, 'shares': shares, 'sold': False}}
            cash = self.initial_capital
            transactions = []
            portfolio_value_history = []
            
            # Get all dates from price data
            all_dates = sorted(self.price_df.index)
            total_dates = len(all_dates)
            
            # Get all ROCE dates
            roce_dates = sorted([d for d in self.roce_df.columns if pd.notna(d)])
            
            print(f"Processing {total_dates} trading days...")
            
            # For each date, check if any company meets criteria
            for idx, current_date in enumerate(all_dates):
                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{total_dates} dates ({100*(idx+1)/total_dates:.1f}%)")
                # Check if we need to sell any stocks (based on holding period)
                # Sell after 1, 2, 3, 4, or 5 years (first opportunity after each milestone)
                companies_to_sell = []
                for company, position in portfolio.items():
                    # Skip if already marked for sale
                    if position.get('sold', False):
                        continue
                    
                    entry_date = position['entry_date']
                    holding_days = (current_date - entry_date).days
                    
                    # Check if we've reached any exit milestone (1, 2, 3, 4, or 5 years)
                    for exit_years in [1, 2, 3, 4, 5]:
                        days_for_exit = exit_years * 365
                        # Allow some tolerance (within 5 days after the milestone)
                        if days_for_exit <= holding_days < days_for_exit + 5:
                            companies_to_sell.append((company, exit_years))
                            break
                
                # Execute sells
                for company, holding_period in companies_to_sell:
                    if company in portfolio:
                        position = portfolio[company]
                        # Get current price
                        current_price, price_date = self.get_closest_price(company, current_date, max_days=0)
                        if current_price is None:
                            # Use last available price
                            prices = self.price_df[company].dropna()
                            if len(prices) > 0:
                                last_price = prices[prices.index <= current_date]
                                if len(last_price) > 0:
                                    current_price = float(last_price.iloc[-1])
                                    price_date = last_price.index[-1]
                                else:
                                    continue
                            else:
                                continue
                        
                        shares = position['shares']
                        sell_value = shares * current_price
                        cash += sell_value
                        
                        transactions.append({
                            'date': price_date,
                            'company': company,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': sell_value,
                            'holding_period_years': holding_period
                        })
                        
                        position['sold'] = True
                        del portfolio[company]
                
                # Check for new buy signals
                # Check monthly (on first trading day of each month) or when new ROCE data might be available
                # This ensures we catch signals soon after annual ROCE data is published
                should_check = False
                if current_date.day <= 5:  # Check in first 5 days of month
                    should_check = True
                else:
                    # Also check within 30 days of any ROCE reporting date
                    for roce_date in roce_dates:
                        if abs((current_date - roce_date).days) <= 30:
                            should_check = True
                            break
                
                if not should_check:
                    continue
                
                # Check each company for buy signal
                for company in self.companies:
                    # Skip if already in portfolio
                    if company in portfolio:
                        continue
                    
                    # Skip if portfolio is full
                    if len(portfolio) >= self.max_stocks:
                        continue
                    
                    # Check if ROCE is increasing for 'years' years
                    if self.check_roce_increasing(company, current_date, years):
                        # Get price
                        price, price_date = self.get_closest_price(company, current_date, max_days=15)
                        
                        if price is None:
                            continue
                        
                        # Calculate position size
                        position_value = self.initial_capital * self.weight_per_stock
                        shares = position_value / price
                        
                        # Check if we have enough cash
                        if cash >= position_value:
                            cash -= position_value
                            portfolio[company] = {
                                'entry_date': price_date,
                                'entry_price': price,
                                'shares': shares,
                                'sold': False
                            }
                            
                            transactions.append({
                                'date': price_date,
                                'company': company,
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'value': position_value
                            })
                
                # Calculate portfolio value
                portfolio_value = cash
                for company, position in portfolio.items():
                    # Get current price
                    prices = self.price_df[company].dropna()
                    if len(prices) > 0:
                        last_price = prices[prices.index <= current_date]
                        if len(last_price) > 0:
                            current_price = float(last_price.iloc[-1])
                            portfolio_value += position['shares'] * current_price
                
                portfolio_value_history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'num_positions': len(portfolio)
                })
            
            # Sell remaining positions at last available price
            last_date = all_dates[-1]
            for company, position in list(portfolio.items()):
                prices = self.price_df[company].dropna()
                if len(prices) > 0:
                    last_price = prices.iloc[-1]
                    if pd.notna(last_price) and last_price > 0:
                        sell_value = position['shares'] * float(last_price)
                        cash += sell_value
                        
                        transactions.append({
                            'date': last_date,
                            'company': company,
                            'action': 'SELL',
                            'price': float(last_price),
                            'shares': position['shares'],
                            'value': sell_value,
                            'holding_period_years': (last_date - position['entry_date']).days / 365.25
                        })
            
            # Calculate final metrics
            final_value = cash
            total_return = (final_value - self.initial_capital) / self.initial_capital * 100
            
            # Calculate CAGR
            if len(portfolio_value_history) > 0:
                start_date = portfolio_value_history[0]['date']
                end_date = portfolio_value_history[-1]['date']
                years_elapsed = (end_date - start_date).days / 365.25
                
                if years_elapsed > 0 and final_value > 0:
                    cagr = ((final_value / self.initial_capital) ** (1 / years_elapsed) - 1) * 100
                else:
                    cagr = 0.0
            else:
                years_elapsed = 0
                cagr = 0.0
            
            # Create results dataframe
            transactions_df = pd.DataFrame(transactions)
            portfolio_history_df = pd.DataFrame(portfolio_value_history)
            
            results[years] = {
                'total_return_pct': total_return,
                'cagr_pct': cagr,
                'years_elapsed': years_elapsed,
                'final_value': final_value,
                'initial_capital': self.initial_capital,
                'transactions': transactions_df,
                'portfolio_history': portfolio_history_df,
                'num_transactions': len(transactions_df),
                'num_buys': len(transactions_df[transactions_df['action'] == 'BUY']),
                'num_sells': len(transactions_df[transactions_df['action'] == 'SELL'])
            }
            
            print(f"\nResults for {years} year(s) criteria:")
            print(f"  Initial Capital: ₹{self.initial_capital:,.2f}")
            print(f"  Final Value: ₹{final_value:,.2f}")
            print(f"  Total Return: {total_return:.2f}%")
            print(f"  CAGR: {cagr:.2f}%")
            print(f"  Years Elapsed: {years_elapsed:.2f}")
            print(f"  Total Transactions: {len(transactions_df)}")
            print(f"  Buy Signals: {len(transactions_df[transactions_df['action'] == 'BUY'])}")
            print(f"  Sell Signals: {len(transactions_df[transactions_df['action'] == 'SELL'])}")
        
        return results
    
    def generate_graphs(self, results):
        """
        Generate graphs for each strategy showing portfolio value over time
        
        Parameters:
        -----------
        results: dict
            Results dictionary from run_backtest
        """
        if not MATPLOTLIB_AVAILABLE:
            print("\nWarning: matplotlib not available. Skipping graph generation.")
            print("Install matplotlib to generate graphs: pip install matplotlib")
            return
        
        print("\n" + "="*60)
        print("Generating graphs...")
        print("="*60)
        
        # Create figure with subplots for each strategy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ROCE Backtest - Portfolio Value Over Time', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (years, result) in enumerate(sorted(results.items())):
            ax = axes[idx]
            
            portfolio_history = result['portfolio_history']
            if len(portfolio_history) > 0:
                portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
                portfolio_history = portfolio_history.sort_values('date')
                
                # Plot portfolio value
                ax.plot(portfolio_history['date'], portfolio_history['portfolio_value'], 
                       linewidth=2, label=f'{years} Year(s) ROCE Increasing', color=plt.cm.tab10(idx))
                
                # Add initial capital line
                ax.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                          linewidth=1, alpha=0.7, label='Initial Capital')
                
                # Formatting
                ax.set_title(f'{years} Year(s) ROCE Increasing Strategy\n'
                           f'CAGR: {result["cagr_pct"]:.2f}% | '
                           f'Total Return: {result["total_return_pct"]:.2f}%', 
                           fontsize=11, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Portfolio Value (₹)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{years} Year(s) ROCE Increasing Strategy', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('backtest_portfolio_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: backtest_portfolio_comparison.png")
        
        # Create individual graphs for each strategy
        for years, result in results.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            portfolio_history = result['portfolio_history']
            if len(portfolio_history) > 0:
                portfolio_history['date'] = pd.to_datetime(portfolio_history['date'])
                portfolio_history = portfolio_history.sort_values('date')
                
                # Plot portfolio value
                ax.plot(portfolio_history['date'], portfolio_history['portfolio_value'], 
                       linewidth=2.5, color='#2E86AB', label='Portfolio Value')
                
                # Add initial capital line
                ax.axhline(y=self.initial_capital, color='#A23B72', linestyle='--', 
                          linewidth=2, alpha=0.8, label='Initial Capital')
                
                # Add annotations for key metrics
                final_value = result['final_value']
                ax.text(0.02, 0.98, 
                       f'Initial Capital: ₹{self.initial_capital:,.0f}\n'
                       f'Final Value: ₹{final_value:,.0f}\n'
                       f'Total Return: {result["total_return_pct"]:.2f}%\n'
                       f'CAGR: {result["cagr_pct"]:.2f}%\n'
                       f'Years: {result["years_elapsed"]:.2f}\n'
                       f'Buy Signals: {result["num_buys"]}\n'
                       f'Sell Signals: {result["num_sells"]}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8))
                
                # Formatting
                ax.set_title(f'ROCE Backtest - {years} Year(s) Increasing Strategy', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=11)
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                filename = f'backtest_strategy_{years}years.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filename}")
                plt.close()
        
        print("Graph generation completed!")

def main():
    # Initialize backtest
    backtest = ROCEBacktest(
        roce_file='ROCE.csv',
        price_file='close_price.csv',
        initial_capital=10000000,  # 1,00,00,000 (updated by user)
        max_stocks=200,
        weight_per_stock=0.005  # 0.5%
    )
    
    # Run backtest for different year criteria
    results = backtest.run_backtest(years_list=[1, 2, 3, 4])
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    # Prepare summary data
    summary_data = []
    
    for years, result in sorted(results.items()):
        # Save transactions
        result['transactions'].to_csv(f'transactions_{years}years.csv', index=False)
        
        # Save portfolio history
        result['portfolio_history'].to_csv(f'portfolio_history_{years}years.csv', index=False)
        
        # Add to summary
        summary_data.append({
            'Years_ROCE_Increasing': years,
            'Initial_Capital': result['initial_capital'],
            'Final_Value': result['final_value'],
            'Total_Return_Pct': result['total_return_pct'],
            'CAGR_Pct': result['cagr_pct'],
            'Years_Elapsed': result['years_elapsed'],
            'Num_Transactions': result['num_transactions'],
            'Num_Buys': result['num_buys'],
            'Num_Sells': result['num_sells']
        })
    
    # Save comprehensive summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('backtest_summary.csv', index=False)
    
    print("\nSummary saved to backtest_summary.csv")
    print("\n" + "="*60)
    print("SUMMARY OF ALL STRATEGIES")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # Generate graphs
    backtest.generate_graphs(results)
    
    print("\n" + "="*60)
    print("Backtest completed! Results saved to:")
    print("  - transactions_Xyears.csv (for each year criteria)")
    print("  - portfolio_history_Xyears.csv (for each year criteria)")
    print("  - backtest_summary.csv (summary with CAGR for all criteria)")
    print("  - backtest_portfolio_comparison.png (comparison of all strategies)")
    print("  - backtest_strategy_Xyears.png (individual graph for each strategy)")
    print("="*60)

if __name__ == '__main__':
    main()

