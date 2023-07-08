import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pandas_output_setting():
    """Set pandas _output display setting"""
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", None)
    ##pd.set_option('display.max_columns', 500)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", None)
    pd.options.mode.chained_assignment = None  # default='warn'

def price_from_sqrtpricex96(token0_decimal, token1_decimal, sqrtpricex96):
    price = (sqrtpricex96/(2**96))**2
    shifted_price = price / (10**token1_decimal/10**token0_decimal)
    inverted_price = 1/shifted_price
    return inverted_price

def add_price_impact_to_df(df):
    df['block_time'] = pd.to_datetime(df['block_time'])
    df.sort_values('block_time', ascending=True, inplace=True)

    # Calculate price impact    
    df['abs_actual_amount0'] = np.where(df['token_bought_symbol'] == 'GNS', df['token_bought_amount_raw'].abs()/10**gns_decimal, 
                                        df['token_sold_amount_raw'].abs()/10**dai_decimal)
    df['abs_actual_amount1'] = np.where(df['token_bought_symbol'] == 'GNS', df['token_sold_amount_raw'].abs()/10**dai_decimal, 
                                        df['token_bought_amount_raw'].abs()/10**gns_decimal)
    df['paid_price'] = df['abs_actual_amount1']/df['abs_actual_amount0']
    df['post_swap_price'] = df['sqrtPriceX96'].apply(lambda x: price_from_sqrtpricex96(dai_decimal, gns_decimal, x))
    df['pre_swap_price'] = df['post_swap_price'].shift()
    df['absolute_price_diff'] = df['post_swap_price'] - df['pre_swap_price']
    df['relative_price_diff'] = (df['paid_price']-df['pre_swap_price'])/df['pre_swap_price']*100
    
    return df

# Control panel
graph_switch = True
pandas_output_setting()
pd.options.display.float_format = '{:.10f}'.format
dai_decimal = 18 # on polygon
gns_decimal = 18 # on polygon

# Import data
base_dir = os.getcwd()
data_dir = base_dir + '/data/'
output_dir = base_dir + '/output/'

df1 = pd.read_csv(data_dir + 'gns_dai_trading_price_impact_calc_polygon_part1.csv', sep=',', engine='python')
df2 = pd.read_csv(data_dir + 'gns_dai_trading_price_impact_calc_polygon_part2.csv', sep=',', engine='python')
df3 = pd.read_csv(data_dir + 'gns_dai_trading_price_impact_calc_polygon_part3.csv', sep=',', engine='python')
df_list = [df1, df2, df3]

for curr_df in df_list:
    curr_df = add_price_impact_to_df(curr_df)

print('df1')
print(df1)
print('df2')
print(df2)
print('df3')
print(df3)
    
if graph_switch:
    # Combine all three df's so I can visualize them all at once with more data points
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    # Only retain those that bought GNS
    df_combined = df_combined[df_combined['token_bought_symbol'] == 'GNS']
    x_var = 'amount_usd'
    y_var = 'relative_price_diff'
    x_label = 'USD amount ($) of trades that bought GNS'
    y_label = 'Price impact (%)'
    plt.scatter(df_combined[x_var], df_combined[y_var])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Scatter plot between {x_label} and \n{y_label.lower()} on Uniswap-V3 DAI-GNS pool')
    plt.show()
