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
graph_switch = True
arbitrum_trading_switch = True
arbitrum_lp_switch = True
polygon_trading_switch = True
polygon_lp_switch = True
pandas_output_setting()
pd.options.display.float_format = '{:.10f}'.format
start_date = pd.to_datetime('2023-01-06')
end_date = pd.to_datetime('2023-06-06')
token_label = 'GNS'
trading_vol_threshold = 1000 # USD

# Import data
base_dir = os.getcwd()
data_dir = base_dir + '/data/'
output_dir = base_dir + '/output/'

df_token_symbol = pd.read_csv(data_dir + 'erc20_token_symbol_v1.csv', sep=',', engine='python')
df_token_price_arb = pd.read_csv(data_dir + 'erc20_token_price_arb_v0.csv', sep=',', engine='python')
df_token_price_polygion = pd.read_csv(data_dir + 'erc20_token_price_polygon_v0.csv', sep=',', engine='python')

df_lp_pool_created_arb = pd.read_csv(data_dir + 'gns_lp_pool_created_arb_v1.csv', sep=',', engine='python')
df_lp_pool_created_polygon = pd.read_csv(data_dir + 'gns_lp_pool_created_polygon_v1.csv', sep=',', engine='python')

df_lp_add_remove_arb = pd.read_csv(data_dir + 'gns_lp_add_remove_arb_v0.csv', sep=',', engine='python')

df_lp_add_remove_polygon = pd.read_csv(data_dir + 'gns_lp_add_remove_polygon_v0.csv', sep=',', engine='python')

df_trading_arb = pd.read_csv(data_dir + 'gns_trading_arb_v0.csv', sep=',', engine='python')
df_trading_polygon = pd.read_csv(data_dir + 'gns_trading_polygon_v0.csv', sep=',', engine='python')

# Core functions
def create_token_pair(row, token0_label_varname, token1_label_varname):
    if isinstance(row[token0_label_varname], str) and len(row[token0_label_varname]) >= 1:
        if isinstance(row[token1_label_varname], str) and len(row[token1_label_varname]) >= 1:
                    tokens = [row[token0_label_varname], row[token1_label_varname]]
                    tokens.sort()  # Sort tokens alphabetically
                    return '-'.join(tokens)  # Combine tokens with hyphen
        else:
            return ''
    else:
        return ''

def process_trading_data(df, network) -> pd.DataFrame:
    """
        1) Fill missing data on `token_bought_symbol` and `token_sold_symbol` columns,
        2) Fill missing data on `token_pair` column
        df: input df that has standard trading data
        network: network of blockchain, i.e., 'arbitrum', 'polygon', etc
    """
    # Remove columns that I will be replacing
    df = df.drop(['token_bought_symbol', 'token_sold_symbol', 'token_pair', 'token_bought_amount', 'token_sold_amount'], axis=1)
    # Create token symbol mapper
    
    df_token_mapper = df_token_symbol.copy()
    df_token_mapper = df_token_mapper[df_token_mapper['blockchain'] == network]
    assert len(df_token_mapper) > 0
    # Mapping address and extract token symbol in a new column
    df["token_bought_symbol"] = df["token_bought_address"].map(df_token_mapper.set_index("contract_address")["symbol"])
    df["token_sold_symbol"] = df["token_sold_address"].map(df_token_mapper.set_index("contract_address")["symbol"])
    df["token_pair"] = df.apply(create_token_pair, args=('token_bought_symbol', 'token_sold_symbol', ), axis=1)
    # Mapping address and extract token decimal in a new column
    df['token_bought_amount_raw'] = df['token_bought_amount_raw'].astype(float)
    df['token_sold_amount_raw'] = df['token_sold_amount_raw'].astype(float)
    df["token_bought_amount"] = df['token_bought_amount_raw'].div(10**(df["token_bought_address"].map(df_token_mapper.set_index("contract_address")["decimals"])))
    df["token_sold_amount"] = df['token_sold_amount_raw'].div(10**(df["token_sold_address"].map(df_token_mapper.set_index("contract_address")["decimals"])))
    # Mapping with the token pair data to extract the identity of token0 or token1 for token bought and token sold
    # Joining the dataframes based on 'pair_address'
    if network.lower() == 'arbitrum':
        df_merged = df.merge(df_lp_pool_created_arb, left_on='project_contract_address', right_on='pair_address', how='left')
    elif network.lower() == 'polygon':
        df_merged = df.merge(df_lp_pool_created_polygon, left_on='project_contract_address', right_on='pair_address', how='left')
    # Applying the logic to combine the amounts
    df_merged['token0_bought_amount'] = df_merged.apply(
        lambda row: row['token_bought_amount'] if row['token_bought_symbol'] == row['t0_symbol'] else 0, axis=1)
    df_merged['token1_bought_amount'] = df_merged.apply(
        lambda row: row['token_bought_amount'] if row['token_bought_symbol'] == row['t1_symbol'] else 0, axis=1)
    df_merged['token0_sold_amount'] = df_merged.apply(
        lambda row: row['token_sold_amount'] if row['token_sold_symbol'] == row['t0_symbol'] else 0, axis=1)
    df_merged['token1_sold_amount'] = df_merged.apply(
        lambda row: row['token_sold_amount'] if row['token_sold_symbol'] == row['t1_symbol'] else 0, axis=1)    
    assert len(df) == len(df_merged)
    return df_merged

def process_lp_add_remove_data(df, network) -> pd.DataFrame:
    """
        1) Fill missing data on `token0` and `token1` columns,
        2) Create `token_pair` column
        df: input df that has standard LP adding/removing data
        network: network of blockchain, i.e., 'arbitrum', 'polygon', etc
    """
    # Remove columns that I will be replacing
    df = df.drop(['token0', 'token1'], axis=1)
    # Create token symbol mapper
    df_token_mapper = df_token_symbol.copy()
    df_token_mapper = df_token_mapper[df_token_mapper['blockchain'] == network]
    assert len(df_token_mapper) > 0    
    # Mapping address and extract token symbol in a new column
    df["token0"] = df["token0_address"].map(df_token_mapper.set_index("contract_address")["symbol"])
    df["token1"] = df["token1_address"].map(df_token_mapper.set_index("contract_address")["symbol"])
    df["token_pair"] = df.apply(create_token_pair, args=('token0', 'token1',), axis=1)    
    return df   

def descriptive_stats_trading(df, network):
    # Aggregate
    agg_usd = df['amount_usd'].resample('D').sum()
    # Descriptive analytics
    print(f'Between {start_date} and {end_date}:')
    print(f'Daily trading amount (in USD) involved with {token_label} on {network.upper()}:')
    print(df['amount_usd'].describe())
    print(df['amount_usd'].info())
    print(f'Trading volume last 30-day: {agg_usd[-30:].sum()}')
    print(f'Trading volume last 14-day: {agg_usd[-14:].sum()}')
    print(f'Trading volume last 7-day: {agg_usd[-7:].sum()}')
    print(f'Trading volume last 24-hr: {agg_usd[-1]}')

    # Aggregate by "token_pair", and summing "amount_usd"
    df_agg = df.groupby('token_pair').agg(amount_usd=('amount_usd', 'sum')).reset_index().sort_values('amount_usd', ascending=False)
    # Apply trading vol threshold
    print(f"Active trading pairs (threshold at ${trading_vol_threshold} USD): {df_agg[df_agg['amount_usd']>=trading_vol_threshold]['token_pair'].tolist()}")

def create_histogram(df, varname, bin_number, title='Graph title', x_label='x-axis', y_label='y-axis'):
    # Extract the column values as a list
    column_values = df[varname].tolist()

    # Create the histogram
    plt.hist(column_values, bins=bin_number)  # Adjust the number of bins as needed

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the histogram
    plt.show()

def descriptive_stats_lp_add_remove(df, network):  
    token_pair_list = df['token_pair'].unique()
    
    for token_pair in token_pair_list:
        _df = df.copy()
        _df = _df[_df['token_pair'] == token_pair]  
        #_df['day'] = pd.to_datetime(_df['day'])     
        token0 = str(_df['token0'].iloc[-1])
        token1 = str(_df['token1'].iloc[-1])
        # Set up full date range over the rows by increasing by 1 day per row
        index_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df_new = pd.DataFrame(index=index_range)
        # Merge the original DataFrame with the new DataFrame
        df_new = df_new.merge(_df, left_index=True, right_on='day', how='left')
        df_new[['unique_minters', 'unique_burners', 'total_0_added', 'total_0_removed', 'total_1_added',
                'total_1_removed']] = df_new[['unique_minters', 'unique_burners', 'total_0_added', 'total_0_removed', 'total_1_added',
                'total_1_removed']].fillna(0)
        df_new = df_new.fillna(np.nan)
        df_new = df_new.reset_index(drop=True)
        
        # Descriptive analytics
        print(f'Token pair: {token_pair}, token0: {token0}, token1: {token1}')
        print(f'Between {start_date} and {end_date}:')
        print(f'Daily {token0} added to LP pool ({token_pair}) on {network.upper()}:')
        print(df['total_0_added'].describe())
        print(df['total_0_added'].info())
        print(f'LP volume added last 30-day: {df_new["total_0_added"].tail(30).sum()}')
        print(f'LP volume added last 14-day: {df_new["total_0_added"].tail(14).sum()}')
        print(f'LP volume added last 7-day: {df_new["total_0_added"].tail(7).sum()}')
        print(f'LP volume added last 24-hr: {df_new["total_0_added"].tail(1).sum()}')
        print()
        print(f'Between {start_date} and {end_date}:')
        print(f'Daily {token0} removed from the LP pool ({token_pair}) on {network.upper()}:')
        print(df['total_0_removed'].describe())
        print(df['total_0_removed'].info())
        print(f'LP volume removed last 30-day: {df_new["total_0_removed"].tail(30).sum()}')
        print(f'LP volume removed last 14-day: {df_new["total_0_removed"].tail(14).sum()}')
        print(f'LP volume removed last 7-day: {df_new["total_0_removed"].tail(7).sum()}')
        print(f'LP volume removed last 24-hr: {df_new["total_0_removed"].tail(1).sum()}')
        print()
        print(f'Between {start_date} and {end_date}:')
        print(f'Daily {token1} added to the LP pool ({token_pair}) on {network.upper()}:')
        print(df['total_1_added'].describe())
        print(df['total_1_added'].info())
        print(f'LP volume added last 30-day: {df_new["total_1_added"].tail(30).sum()}')
        print(f'LP volume added last 14-day: {df_new["total_1_added"].tail(14).sum()}')
        print(f'LP volume added last 7-day: {df_new["total_1_added"].tail(7).sum()}')
        print(f'LP volume added last 24-hr: {df_new["total_1_added"].tail(1).sum()}')
        print()
        print(f'Between {start_date} and {end_date}:')
        print(f'Daily {token1} removed from the LP pool ({token_pair}) on {network.upper()}:')
        print(df['total_1_added'].describe())
        print(df['total_1_added'].info())
        print(f'LP volume removed last 30-day: {df_new["total_1_added"].tail(30).sum()}')
        print(f'LP volume removed last 14-day: {df_new["total_1_added"].tail(14).sum()}')
        print(f'LP volume removed last 7-day: {df_new["total_1_added"].tail(7).sum()}')
        print(f'LP volume removed last 24-hr: {df_new["total_1_added"].tail(1).sum()}')
      
    # Aggregate df on LP adding and removing
    df_agg = df.groupby('token_pair').agg({'token0': 'last', 'token1': 'last', 'total_0_added': 'sum', 'total_0_removed': 'sum',
                                           'total_1_added': 'sum', 'total_1_removed': 'sum'}).reset_index()
    df_agg['lp_total_0_net'] = df_agg['total_0_added'] + df_agg['total_0_removed'] 
    df_agg['lp_total_1_net'] = df_agg['total_1_added'] + df_agg['total_1_removed'] 
    df_agg['date'] = end_date
    
    print('Aggregate total of LP activities:')
    print(df_agg)

# ////////////////////
# ////////////////////
if arbitrum_trading_switch:
    # GNS on Arbitrum Uniswap V3 pools - trading
    print('Number of LP pairs with GNS created on Arbitrum:', len(df_lp_pool_created_arb))
    df_trading_arb = process_trading_data(df_trading_arb, 'arbitrum')

    # Date manipulation
    df_trading_arb['block_date'] = pd.to_datetime(df_trading_arb['block_date'])
    df_trading_arb.sort_values('block_date', ascending=True, inplace=True)
    # Set 'block_date' column as the DataFrame index
    df_trading_arb.set_index('block_date', inplace=True)
    # Set date range
    df_trading_arb = df_trading_arb[(df_trading_arb.index >= start_date) & (df_trading_arb.index <= end_date)]

    # Aggregate the total trading amount by date
    total_trading_amount = df_trading_arb['amount_usd'].resample('D').sum()

    if graph_switch:
        # Create a line plot to visualize the time-series data
        plt.plot(total_trading_amount.index, total_trading_amount.values, label='Total Trading Amount')
        plt.xlabel('Date')
        plt.ylabel('Amount traded in USD')
        plt.title('Daily trading with GNS-pair on Uniswap-V3 on Arbitrum')
        plt.legend()
        plt.show()

    # Descriptive analytics
    print('///')
    print(f'Trading on Arbitrum including all trades with {token_label}:')
    descriptive_stats_trading(df=df_trading_arb, network='arbitrum')

    print('///')
    print(f'Trading on Arbitrum including only trades that sold {token_label}:')
    df_trading_arb_only_sold_gns = df_trading_arb.copy()
    df_trading_arb_only_sold_gns = df_trading_arb_only_sold_gns[df_trading_arb_only_sold_gns['token_sold_symbol'] == token_label.upper()]
    descriptive_stats_trading(df=df_trading_arb_only_sold_gns, network='arbitrum')

    print('///')
    print(f'Trading on Arbitrum including only trades that bought {token_label}:')
    df_trading_arb_only_bought_gns = df_trading_arb.copy()
    df_trading_arb_only_bought_gns = df_trading_arb_only_bought_gns[df_trading_arb_only_bought_gns['token_bought_symbol'] == token_label.upper()]
    descriptive_stats_trading(df=df_trading_arb_only_bought_gns, network='arbitrum')

    if graph_switch:
        # Create histogram for "amount_usd" frequency
        create_histogram(df_trading_arb, 'amount_usd', 30, title='Frequency of trade size on Arbitrum Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')
        create_histogram(df_trading_arb_only_sold_gns, 'amount_usd', 30, title='Frequency of trade size (only those sold GNS) on Arbitrum Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')
        create_histogram(df_trading_arb_only_bought_gns, 'amount_usd', 30, title='Frequency of trade size (only those bought GNS) on Arbitrum Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')

# ////////////////////
# ////////////////////
if arbitrum_lp_switch:
    # GNS on Arbitrum Uniswap V3 pools - LP provision
    df_lp_add_remove_arb = process_lp_add_remove_data(df_lp_add_remove_arb, 'arbitrum')
    # Date manipulation
    df_lp_add_remove_arb['day'] = pd.to_datetime(df_lp_add_remove_arb['day'])
    df_lp_add_remove_arb.sort_values('day', ascending=True, inplace=True)
    # Set 'day' column as the DataFrame index
    df_lp_add_remove_arb.set_index('day', inplace=True)
    # Set date range
    df_lp_add_remove_arb = df_lp_add_remove_arb[(df_lp_add_remove_arb.index >= start_date) & (df_lp_add_remove_arb.index <= end_date)]
    descriptive_stats_lp_add_remove(df_lp_add_remove_arb, network='arbitrum')

# ////////////////////
# ////////////////////
if polygon_trading_switch:
    # GNS on Polygon Uniswap V3 pools
    print('Number of LP pairs with GNS created on Polygon:', len(df_lp_pool_created_polygon))
    df_trading_polygon = process_trading_data(df_trading_polygon, 'polygon')

    # Date manipulation
    df_trading_polygon['block_date'] = pd.to_datetime(df_trading_polygon['block_date'])
    df_trading_polygon.sort_values('block_date', ascending=True, inplace=True)
    # Set 'block_date' column as the DataFrame index
    df_trading_polygon.set_index('block_date', inplace=True)
    # Set date range
    df_trading_polygon = df_trading_polygon[(df_trading_polygon.index >= start_date) & (df_trading_polygon.index <= end_date)]

    # Aggregate the total trading amount by date
    total_trading_amount = df_trading_polygon['amount_usd'].resample('D').sum()

    if graph_switch:
        # Create a line plot to visualize the time-series data
        plt.plot(total_trading_amount.index, total_trading_amount.values, label='Total Trading Amount')
        plt.xlabel('Date')
        plt.ylabel('Amount traded in USD')
        plt.title('Daily trading with GNS-pair on Uniswap-V3 on Polygon')
        plt.legend()
        plt.show()

    # Descriptive analytics
    print('///')
    print(f'Trading on Polygon including all trades with {token_label}:')
    descriptive_stats_trading(df=df_trading_polygon, network='polygon')

    print('///')
    print(f'Trading on Polygon including only trades that sold {token_label}:')
    df_trading_polygon_only_sold_gns = df_trading_polygon.copy()
    df_trading_polygon_only_sold_gns = df_trading_polygon_only_sold_gns[df_trading_polygon_only_sold_gns['token_sold_symbol'] == token_label.upper()]
    descriptive_stats_trading(df=df_trading_polygon_only_sold_gns, network='polygon')

    print('///')
    print(f'Trading on Polygon including only trades that bought {token_label}:')
    df_trading_polygon_only_bought_gns = df_trading_polygon.copy()
    df_trading_polygon_only_bought_gns = df_trading_polygon_only_bought_gns[df_trading_polygon_only_bought_gns['token_bought_symbol'] == token_label.upper()]
    descriptive_stats_trading(df=df_trading_polygon_only_bought_gns, network='polygon')

    if graph_switch:
        # Create histogram for "amount_usd" frequency
        create_histogram(df_trading_polygon, 'amount_usd', 30, title='Frequency of trade size (only those sold GNS) on Polygon Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')
        create_histogram(df_trading_polygon_only_sold_gns, 'amount_usd', 30, title='Frequency of trade size (only those sold GNS) on Polygon Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')
        create_histogram(df_trading_polygon_only_bought_gns, 'amount_usd', 30, title='Frequency of trade size (only those sold GNS) on Polygon Uniswap-V3 pools', x_label='Trade size (USD)', y_label='Frequency')

# ////////////////////
# ////////////////////
if polygon_lp_switch:
    # GNS on Polygon Uniswap V3 pools - LP provision
    df_lp_add_remove_polygon = process_lp_add_remove_data(df_lp_add_remove_polygon, 'polygon')
    # Date manipulation
    df_lp_add_remove_polygon['day'] = pd.to_datetime(df_lp_add_remove_polygon['day'])
    df_lp_add_remove_polygon.sort_values('day', ascending=True, inplace=True)
    # Set 'day' column as the DataFrame index
    df_lp_add_remove_polygon.set_index('day', inplace=True)
    # Set date range
    df_lp_add_remove_polygon = df_lp_add_remove_polygon[(df_lp_add_remove_polygon.index >= start_date) & (df_lp_add_remove_polygon.index <= end_date)]
    descriptive_stats_lp_add_remove(df_lp_add_remove_polygon, network='polygon')





