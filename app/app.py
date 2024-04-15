import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yfinance as yf
import itertools
import sys
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config
from finrl.config import INDICATORS
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import itertools

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

folder_path = 'results/'  # Your folder path
file_path = os.path.join(folder_path, 'data.csv')
x = 1

# Define your model function
def trade(initial_investment, trade_start_date, trade_end_date):
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2020-01-01'
    buy_stocks = []  # List to store stocks to buy
    sell_stocks = []  # List to store stocks to sell
    processed_full = pd.read_csv('processed_full.csv')
    processed_full = processed_full.drop(processed_full.columns[0], axis=1)
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, str(trade_start_date), str(trade_end_date))
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": int(initial_investment),
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    if_using_ppo = True
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    trained_model = PPO.load('ppo_agent.zip')
    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym)
    # Iterate over each row in the DataFrame
    df = df_actions_ppo
    df1 = df_account_value_ppo
    ROI = ((df1['account_value'].iloc[-1] - df1['account_value'].iloc[0]) / df1['account_value'].iloc[0]) * 100

    # Putting index as the first column
    df.reset_index(inplace=True)

    # Adding a new index
    df['new_index'] = range(1, len(df) + 1)

    # Setting the new index
    df.set_index('new_index', inplace=True)

    for index, row in df.iterrows():
        date = row['date']
        # Iterate over the stocks and check their values
        for column in df.columns[1:]:
            stock_value = row[column]

            if stock_value > 0:
                buy_stocks.append(f"On {date}, buy {abs(stock_value)} shares of {column}")
            elif stock_value < 0:
                sell_stocks.append(f"On {date}, sell {abs(stock_value)} shares of {column}")

    if if_using_ppo:
        print("\n ppo:")
        perf_stats_all_ppo = backtest_stats(account_value=df_account_value_ppo)
        perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)
    return [buy_stocks, sell_stocks, ROI]


def trade2(initial_investment, trade_start_date):
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2020-01-01'
    TRADE_START_DATE1 = str(trade_start_date)
    TRADE_START_DATE_datetime = datetime.strptime(TRADE_START_DATE1, '%Y-%m-%d')
    TRADE_END_DATE_datetime = TRADE_START_DATE_datetime + timedelta(days=60)
    TRADE_END_DATE1 = TRADE_END_DATE_datetime.strftime('%Y-%m-%d')

    TRADE_END_DATE2 = TRADE_START_DATE1
    TRADE_END_DATE_datetime = datetime.strptime(TRADE_END_DATE2, '%Y-%m-%d')
    TRADE_START_DATE_datetime = TRADE_END_DATE_datetime - timedelta(days=60)
    TRADE_START_DATE2 = TRADE_START_DATE_datetime.strftime('%Y-%m-%d')

    top_5_buy = []
    top_5_sell = []
    top_5_buy_list = []
    top_5_sell_list = []

    processed_full = pd.read_csv('processed_full.csv')
    processed_full = processed_full.drop(processed_full.columns[0], axis=1)
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade1 = data_split(processed_full, TRADE_START_DATE1, TRADE_END_DATE1)
    trade2 = data_split(processed_full, TRADE_START_DATE2, TRADE_END_DATE2)
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": int(initial_investment),
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    if_using_ppo = True

    trade_list = [trade1, trade2]

    for index, trade in enumerate(trade_list):
        e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
        trained_model = PPO.load('ppo_agent.zip')
        df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
            model=trained_model,
            environment=e_trade_gym)
        # Iterate over each row in the DataFrame
        if index == 0:
            df = df_actions_ppo
            new_df = df.head(2)
            new_df.drop(columns=new_df.columns[0], axis=1, inplace=True)
            new_df = new_df.sum()
            new_df = new_df.sort_values(ascending=False).head(5)
            top_5_buy = new_df.index.tolist()
        else:
            df = df_actions_ppo
            valid_dates = df.index[df.ne(0).any(axis=1)].unique()[-2:]
            last_two_dates_df = df[df.index.isin(valid_dates)]
            last_two_dates_df = last_two_dates_df.drop(columns=top_5_buy, errors='ignore')
            total_shares = last_two_dates_df.sum()
            top_5_to_sell = total_shares.nsmallest(5)
            top_5_sell = top_5_to_sell.index.tolist()

        if if_using_ppo:
            print("\n ppo:")
            perf_stats_all_ppo = backtest_stats(account_value=df_account_value_ppo)
            perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)

    # Load the CSV file containing symbols and names into a DataFrame
    df1 = pd.read_csv('stocks.csv')

    # Create a dictionary mapping symbols to names
    symbol_to_name = df1.set_index('Symbol')['Name'].to_dict()

    for symbol in top_5_buy:
        symbol = symbol.replace('-', '.')
        name = symbol_to_name[symbol]
        symbol = symbol.replace('.TO', '')
        # Format the name with the symbol
        formatted_name = f"{name}({symbol})"
        top_5_buy_list.append(formatted_name)
    
    for symbol in top_5_sell:
        symbol = symbol.replace('-', '.')
        name = symbol_to_name[symbol]
        symbol = symbol.replace('.TO', '')
        # Format the name with the symbol
        formatted_name = f"{name}({symbol})"
        top_5_sell_list.append(formatted_name)

    return [top_5_buy_list, top_5_sell_list]


def perf_index(date1, date2):
    date1 = str(date1)
    date2 = str(date2)
    df = pd.read_csv('gsptse.csv')
    if date1 not in df['date'].values:
        date1 = df.loc[df['date'] > date1, 'date'].min()
    # Check if date2 is present in the DataFrame, if not, find the closest date before it
    if date2 not in df['date'].values:
        date2 = df.loc[df['date'] < date2, 'date'].max()

    filtered_df = df[df['date'].isin([date1, date2])]

    # Extract the close prices for each date
    close_price_date1 = filtered_df.loc[filtered_df['date'] == date1, 'close'].values[0]
    close_price_date2 = filtered_df.loc[filtered_df['date'] == date2, 'close'].values[0]

    # Calculate the percentage change between the close prices
    percentage_change = ((close_price_date2 - close_price_date1) / close_price_date1) * 100
    return percentage_change


def update_csv(y1, y2, file_path):
    # Check if the CSV file exists
    if os.path.exists(file_path):
        # Read the existing CSV file
        existing_data = pd.read_csv(file_path, index_col=0)  # Specify existing index column
        # Drop unnamed columns if they exist
        existing_data = existing_data.loc[:, ~existing_data.columns.str.contains('^Unnamed')]
        # Keep only the last 9 rows
        existing_data = existing_data.tail(9)
        # Create new DataFrame for the new data
        new_data = pd.DataFrame({'y1': [y1], 'y2': [y2]})
        # Append the new data to the existing data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        # Reset index
        updated_data = updated_data.reset_index(drop=True)
        # Write the updated data to the CSV file with index
        updated_data.to_csv(file_path)
    else:
        # Create new DataFrame for the new data
        new_data = pd.DataFrame({'y1': [y1], 'y2': [y2]})
        # Write the new data to a new CSV file with index
        new_data.to_csv(file_path, index=True)


def tab2_content():
    initial_investment = st.number_input('Initial Investment', min_value=10000, value=1000000)
    smin=pd.to_datetime('2017-01-03').date()
    smax=pd.to_datetime('2023-02-27').date()
    svalue=pd.to_datetime('2023-01-01').date()
    emin=pd.to_datetime('2017-03-03').date()
    emax=pd.to_datetime('2023-04-27').date()
    evalue=pd.to_datetime('2023-04-27').date()
    
    start_date = st.date_input('Start Date', 
                           min_value=smin, 
                           max_value=smax, 
                           value=svalue)
    end_date = st.date_input('End Date', 
                         min_value=emin, 
                         max_value=emax, 
                         value=evalue)

    # Perform prediction when 'Predict' button is clicked
    if st.button('Submit'):
        buy_stocks, sell_stocks, ROI = trade(initial_investment, start_date, end_date)
        iperf = perf_index(start_date, end_date)

        # Round ROI and iperf to two decimal places
        rounded_ROI = round(ROI, 2)
        rounded_iperf = round(iperf, 2)
        
        # Display rounded values with increased font size
        st.write(f'<span style="font-size:18px; font-weight:bold;">ROI: {rounded_ROI}%</span>', unsafe_allow_html=True)
        st.write(f'<span style="font-size:18px; font-weight:bold;">Change in TSX index: {rounded_iperf}%</span>', unsafe_allow_html=True)
        st.write(f'Buy_info', buy_stocks)
        st.write(f'Sell_info', sell_stocks)

        y1 = ROI
        y2 = iperf
        update_csv(y1, y2, file_path)
        updated_data = pd.read_csv(file_path, index_col=0)
        a = list(updated_data.index)
        b = list(updated_data['y1'])
        c = list(updated_data['y2'])

        # iperf_change, ROI_list = graph(start_date, end_date)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=a,
            y=c,
            line=dict(color='firebrick', dash='solid'),
            name='Change in TSX index'

        ))

        fig.add_trace(go.Scatter(
            x=a,
            y=b,
            line=dict(color='#F7DC6F', dash='solid', ),
            name='ROI'
        ))

        fig.update_layout(
            title='Comparison of Performance',
            xaxis_title='Dates',
            yaxis_title='% Change'
        )

        # Display the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)


def tab1_content():
    initial_investment = st.number_input('Initial Investment', min_value=100, value=100000)
    
    smin=pd.to_datetime('2017-03-03').date()
    smax=pd.to_datetime('2023-02-27').date()
    svalue=pd.to_datetime('2021-01-01').date()
    start_date = st.date_input('Start Date', 
                           min_value=smin, 
                           max_value=smax,
                           value=svalue)
    
    # Perform prediction when 'Predict' button is clicked
    if st.button('Submit'):

        top_5_buy, top_5_sell = trade2(initial_investment, start_date)
        # Display the top 5 stocks to buy
        st.title('Top 5 Stocks Recommendation')
        # Display the top 5 stocks to buy and sell side by side
        col1, col2 = st.columns(2)

        # Customize styles for better appearance
        with col1:
            st.header('Top 5 Stocks to Buy')
            for i, stock in enumerate(top_5_buy, start=1):
                st.write(f"{i}. {stock}", unsafe_allow_html=True)

        with col2:
            st.header('Top 5 Stocks to Sell')
            for i, stock in enumerate(top_5_sell, start=1):
                st.write(f"{i}. {stock}", unsafe_allow_html=True)

        # Customize font size and colors
        st.markdown(
            """
            <style>
                .stMarkdown > div {
                    font-size: 18px;
                    color: #2E4053;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


def main():
    st.sidebar.header("ANSBot Dashboard")

    # Add a sidebar with tabs
    tab_selection = st.sidebar.radio("Select Tab", ['Top Recommendations', 'Insights'])

    # Display content based on tab selection
    if tab_selection == 'Top Recommendations':
        st.title('ANSBot Recommendation App')
        tab1_content()
    elif tab_selection == 'Insights':
        st.title('ANSBot Financial Insights')
        tab2_content()


if __name__ == '__main__':
    main()
