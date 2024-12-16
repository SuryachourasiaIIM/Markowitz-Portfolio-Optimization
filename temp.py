import os
os.system('pip install yfinance')
os.system('pip install matplotlib')
os.system('pip install seaborn')
os.system('pip install scipy')
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")  # Use the Agg backend
import matplotlib.pyplot as plt

# Function to fetch adjusted close prices
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Portfolio performance: return and volatility
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Negative Sharpe Ratio (to maximize Sharpe Ratio)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

# Minimum Volatility Portfolio
def minimum_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
                      x0=np.ones(num_assets)/num_assets, bounds=bounds, constraints=constraints)
    return result

# Portfolio Optimization
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, x0=np.ones(num_assets)/num_assets, 
                      args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Random Portfolio Generation
def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios=5000):
    num_assets = len(mean_returns)
    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': []}
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
        p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (p_return - risk_free_rate) / p_volatility
        results['Returns'].append(p_return)
        results['Volatility'].append(p_volatility)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)
    return pd.DataFrame(results)

# Generate Efficient Frontier
def generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=100):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    target_volatilities = []

    num_assets = len(mean_returns)
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1], 
                          x0=np.ones(num_assets)/num_assets, bounds=bounds, constraints=constraints)
        target_volatilities.append(result.fun)

    return target_returns, target_volatilities

# Streamlit UI
st.title("Markowitz Portfolio Optimization")

# Sidebar Inputs
st.sidebar.markdown("""
                    <p style='text-align: center; font-size: small; color: grey;'>Created by - Surya Chourasia</p>
                    <hr style='border-top: 1px solid #ccc; margin: 20px 0;'>
                     """,
                     unsafe_allow_html=True)
st.sidebar.header("Portfolio Inputs")
tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN").split(',')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0) / 100

if st.sidebar.button("Run Optimization"):
    # Fetch Data
    data = get_data(tickers, start_date, end_date)
    returns = data.pct_change().dropna()

    # Calculate Metrics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Current Prices and Currencies
    current_prices = {}
    currencies = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")
        if not info.empty:
            current_prices[ticker] = info['Close'].iloc[-1]
            currencies[ticker] = stock.info.get("currency", "N/A")
        else:
            current_prices[ticker] = "N/A"
            currencies[ticker] = "N/A"
            
    # Optimize Portfolio
    optimized_result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
    optimized_weights = optimized_result.x
    opt_return, opt_volatility = portfolio_performance(optimized_weights, mean_returns, cov_matrix)

    # Minimum Variance Portfolio
    min_var_result = minimum_variance_portfolio(mean_returns, cov_matrix)
    min_var_weights = min_var_result.x
    min_var_return, min_var_volatility = portfolio_performance(min_var_weights, mean_returns, cov_matrix)

    # Generate Random Portfolios
    random_portfolios = generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate)

    # Display Optimized Portfolio Weights
    st.subheader("Optimized Portfolio Weights")
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': optimized_weights,
        'Current Price': [f"{currencies[ticker]} {current_prices[ticker]:.2f}" if isinstance(current_prices[ticker], float) else "N/A" for ticker in tickers]
    })
    st.table(weights_df)

    # Optimized Results
    st.subheader("Optimized Portfolio Results")
    st.write(f"**Expected Annual Return:** {opt_return:.2%}")
    st.write(f"**Annual Volatility (Risk):** {opt_volatility:.2%}")
    st.write(f"**Sharpe Ratio:** {(opt_return - risk_free_rate) / opt_volatility:.2f}")

    # Efficient Frontier Chart
    st.subheader("Efficient Frontier")
    target_returns, target_volatilities = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot efficient frontier
    ax.plot(target_volatilities, target_returns, linestyle='--', label='Efficient Frontier')

    # Random portfolios with Sharpe colorbar
    sc = ax.scatter(random_portfolios['Volatility'], random_portfolios['Returns'], 
                     c=random_portfolios['Sharpe'], cmap='viridis', alpha=0.5)
    ax.scatter(opt_volatility, opt_return, c='red', marker='*', s=200, label='Optimized Portfolio')
    ax.scatter(min_var_volatility, min_var_return, c='blue', marker='o', s=100, label='Minimum Variance Portfolio')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Sharpe Ratio')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    st.pyplot(fig)

    # Chart 2: Correlation Matrix
    correlation_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.index)

    # Chart 3: Historical Returns
    st.subheader("Historical Returns")
    cumulative_returns = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_returns.plot(ax=ax)
    ax.set_title("Cumulative Returns of Individual Stocks")
    st.pyplot(fig)

    # Chart 4: Optimized Portfolio Historical Performance
    st.subheader("Optimized Portfolio Historical Performance")
    optimized_cumulative_returns = (returns @ optimized_weights + 1).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    optimized_cumulative_returns.plot(ax=ax, color='green', label="Optimized Portfolio")
    ax.set_title("Optimized Portfolio Cumulative Returns")
    ax.legend()
    st.pyplot(fig)
