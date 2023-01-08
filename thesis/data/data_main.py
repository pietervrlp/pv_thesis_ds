import ta
import numpy as np
import pandas as pd

from data_utils import separate_stock_data


separate_stock_data('raw_data/daily_us.feather')




# single stock case
df_MSFT = pd.read_feather('stock_data/df_MSFT.ftr')
df_MSFT

# import FF research factors
ff5 = pd.read_csv('raw_data/F-F_Research_Data_5_Factors_2x3_daily.CSV', header=0)
mom = pd.read_csv('raw_data/F-F_Momentum_Factor_daily.CSV', header=0)
lt_rev = pd.read_csv('raw_data/F-F_LT_Reversal_Factor_daily.csv', header=0)
st_rev = pd.read_csv('raw_data/F-F_ST_Reversal_Factor_daily.csv', header=0)
ff5 = pd.merge(ff5,mom,'inner')
ff5 = pd.merge(ff5,lt_rev,'inner')
ff5 = pd.merge(ff5,st_rev,'inner')
ff5.loc[:,'Mkt-RF':] = ff5.loc[:,'Mkt-RF':].div(100)
ff5

# merge to stock
df_MSFT = pd.merge(df_MSFT,ff5,'left')
df_MSFT

# add technical indicators
df_MSFT_narm = ta.utils.dropna(df_MSFT)
df_MSFT_all = ta.add_all_ta_features(df_MSFT_narm, 'open', 'high', 'low', 'close', 'volume', fillna = True)
df = df_MSFT_all.reset_index().drop(columns = ['index'])
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d")

