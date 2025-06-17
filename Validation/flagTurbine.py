# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:03:05 2025

@author: erica
"""
import pandas as pd

load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-30/'
metadata = pd.read_csv(f'{load_folder}/turbine_year_metadata.csv')
# Clean and prepare the data
df_clean = metadata[['turbine_id', 'year', 'season', 'initial_len', 'valid_len', 'active_len']].dropna()

# Aggregate by turbine_id over all seasons and years
agg_df = df_clean.groupby('turbine_id').agg(
    total_initial_len=('initial_len','sum'),
    total_valid_len=('valid_len', 'sum'),
    total_active_len=('active_len', 'sum')
).reset_index()

# Compute the average values across all turbines
#avg_valid_len = agg_df['total_valid_len'].mean()
#avg_active_len = agg_df['total_active_len'].mean()
avg_init_len = agg_df['total_initial_len'].mean()
# Define a threshold for flagging (e.g., 25% below average)
threshold = 0.94

# Flag turbines with significant down period (low valid_len)
agg_df['down_period_flag'] = agg_df['total_valid_len'] < (threshold * avg_init_len)

# Flag turbines with lots of outliers (low active_len relative to valid_len)
agg_df['outlier_flag'] = agg_df['total_active_len'] < (threshold * agg_df['total_valid_len'])

# Display flagged turbines
flagged = agg_df[agg_df['down_period_flag'] | agg_df['outlier_flag']]

# Output the result
print("Flagged turbines:\n", flagged[['turbine_id', 'down_period_flag', 'outlier_flag']])
