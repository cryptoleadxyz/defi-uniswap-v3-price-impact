import os
import sys
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

# Control panel
save_switch = False
pandas_output_setting()
pd.options.display.float_format = '{:.10f}'.format

# Import data
base_dir = os.getcwd()
data_dir = base_dir + '/data/'
output_dir = base_dir + '/output/'

df_token_symbol = pd.read_csv(data_dir + 'erc20_token_symbol_v1.csv', sep=',', engine='python')
df_token_price_arb = pd.read_csv(data_dir + 'erc20_token_price_arb_v0.csv', sep=',', engine='python')
df_token_price_polygion = pd.read_csv(data_dir + 'erc20_token_price_polygon_v0.csv', sep=',', engine='python')

df_trading_polygon = pd.read_csv(data_dir + 'gns_trading_polygon_v0.csv', sep=',', engine='python')

# Extract desired segment of trading data for Polygon DAI-GNS pool, then add the sqrtPriceX96 (from etherscan), and save into .csv file
df_trading_polygon_filtered = df_trading_polygon.loc[df_trading_polygon['project_contract_address']=='0xba0216254163b57af68b7161cf824dbadcad61df']
df_trading_polygon_filtered['block_time'] = pd.to_datetime(df_trading_polygon_filtered['block_time'])
df_trading_polygon_filtered.sort_values('block_time', ascending=True, inplace=True)
df_trading_polygon_filtered = df_trading_polygon_filtered.reset_index(drop=True)

# df_trading_polygon_filtered_max = df_trading_polygon_filtered.sort_values('amount_usd', ascending=False)
# print(df_trading_polygon_filtered_max.head(10)) # from this I picked row index 7494, 7158, 6098 to anchor and draw adjacent rows (5 rows before and 5 rows after) from, since these index rows have some of the highest trading amount to buy GNS

row_index = [7494, 7158, 6098]
# The sqrtPriceX96 and liquidity that are manually extracted from polygonscan, and to be added to the selected data below
extra_data = {'7494': [29118655276540118338466163377, 29131781070069152769027783099, 29149695407101763395295242664, 29174677695345134153957608951,
                       29171954255905305739878296150, 28841871226790163736514424950, 28814899014732786436718648962, 28830885500280360444140262157,
                       28785667668731210931920780797, 28810736907321050997172416190, 28822102094975116341577507532],
              '7158': [30915249005609518738295576159, 30918199805276331170742586121, 30911533464961097279818751949, 30930755950021180944402214885,
                       30938522283844017857545509790, 30607889555728969226128388951, 30639634480148023924318380052, 30671874820335882644076673177,
                       30674209589698765843035674344, 30677092375150464676825333980, 30700719031619072591444308553],
              '6098': [42244770699730657080734429259, 42084741551103159162845097434, 42061566116324694967425737328, 41941959526077126313843356987,
                       41897224540960031720382383238, 40117151260988714014462672273, 40359046983865361265465546773, 40353394914332312378422883766,
                       40694456885916981079767975632, 41185146119177792919538966591, 41256267971395614778753239038]}
count = 1
rows_before = 5
rows_after = 5

for row_idx in row_index:
    df = df_trading_polygon_filtered.copy()
    # Calculate the range of indices to select
    start_index = max(0, row_idx - rows_before)
    end_index = min(row_idx + rows_after + 1, len(df))

    # Use iloc to select the rows based on the calculated range
    selected_rows = df.iloc[start_index:end_index]
    
    # Add sqrtPriceX96 column
    selected_rows['sqrtPriceX96'] = extra_data[str(row_idx)]
    
    # Data prep
    # print(selected_rows.dtypes)
    selected_rows['token_bought_amount_raw'] = selected_rows['token_bought_amount_raw'].astype(float)
    selected_rows['token_sold_amount_raw'] = selected_rows['token_sold_amount_raw'].astype(float)
    selected_rows['sqrtPriceX96'] = selected_rows['sqrtPriceX96'].astype(float)
    
    # Saving
    if save_switch:
        selected_rows.to_csv(data_dir+f'gns_dai_trading_price_impact_calc_polygon_part{count}.csv', index=False)
        count += 1
    
    
