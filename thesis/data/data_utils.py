import ta
import numpy as np
import pandas as pd


def get_ticker_list(df):
    ticker_list = df.ticker.unique()
    return ticker_list

def separate_stock_data(df):
    ticker_list = df.ticker.unique()
    for ticker in ticker_list:
        data = df.iloc[list(np.where(df['ticker'] == ticker)[0])].reset_index().drop(columns = ['index'])
        data.to_feather('./stock_data/df_{}.ftr'.format(ticker))
        print(ticker, ' finished')
    return ticker_list

def add_market_factors(df):
    ff5 = pd.read_csv('raw_data/F-F_Research_Data_5_Factors_2x3_daily.CSV', header=0)
    mom = pd.read_csv('raw_data/F-F_Momentum_Factor_daily.CSV', header=0)
    lt_rev = pd.read_csv('raw_data/F-F_LT_Reversal_Factor_daily.csv', header=0)
    st_rev = pd.read_csv('raw_data/F-F_ST_Reversal_Factor_daily.csv', header=0)
    ff5 = pd.merge(ff5,mom,'inner')
    ff5 = pd.merge(ff5,lt_rev,'inner')
    ff5 = pd.merge(ff5,st_rev,'inner')
    ff5.loc[:,'Mkt-RF':] = ff5.loc[:,'Mkt-RF':].div(100)
    df = pd.merge(df,ff5,'left')
    return df

def add_all_technical_indicators(df):
    df = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna = True)
    return df

def strategy_staffini(df, ticker):
    df = df[['date','close','open','high','low','volume']]
    df['SMA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA26'] = ta.trend.sma_indicator(df['close'], window=26)
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACDsign'] = ta.trend.macd_signal(df['close'], window=9)
    df['BOLlow'] = ta.volatility.bollinger_lband(df['close'])
    df['BOLup'] = ta.volatility.bollinger_hband(df['close'])
    df = df.dropna().reset_index().drop(columns = ['index'])
    df.to_feather(f'./df_{ticker}_repl.feather')

def strategy_plus(df, ticker):
    df = df[['date','close','open','high','low','volume']]
    df['SMA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA26'] = ta.trend.sma_indicator(df['close'], window=26)
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACDsign'] = ta.trend.macd_signal(df['close'], window=9)
    df['BOLlow'] = ta.volatility.bollinger_lband(df['close'], window=20)
    df['BOLup'] = ta.volatility.bollinger_hband(df['close'], window=20)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['KCH'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['KCL'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
    df['ICHIKOMU_A'] = ta.trend.ichimoku_a(df['high'], df['low'], window1=9, window2=26)
    df['ICHIKOMU_B'] = ta.trend.ichimoku_b(df['high'], df['low'], window2=26, window3=52)
    df['ICHIKOMU_BASE'] = ta.trend.ichimoku_base_line(df['high'], df['low'], window1=9, window2=26)
    df['ICHIKOMU_CONV'] = ta.trend.ichimoku_conversion_line(df['high'], df['low'], window1=9, window2=26)
    df['KAMA'] = ta.momentum.kama(df['close'], window=10, pow1=2, pow2=30)
    df = df.dropna().reset_index().drop(columns = ['index'])
    df.to_feather(f'./df_{ticker}_repl_plus.feather')
