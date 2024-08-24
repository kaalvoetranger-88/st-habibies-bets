#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:51:14 2024

@author: kaalvoetranger

http://tennis-data.co.uk/alldata.php



"""

#%%

import warnings
from pathlib import Path
import pandas as pd
from datetime import datetime

# Ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Directory to save/load CSV files on Desktop
save_dir = Path.home() / "Desktop/Tennis App/datasets"
save_dir.mkdir(parents=True, exist_ok=True)

# Load only Excel files starting with '2' (e.g., 2020.xlsx, 2021.xlsx, etc.)
excel_files = sorted(save_dir.glob('2*.xlsx'))
all_data = pd.concat((pd.read_excel(file) for file in excel_files), ignore_index=True)

# Ensure 'Date' column is a datetime object
all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')

# Sort by the 'Date' column, with oldest match at the top
all_data = all_data.sort_values(by='Date')

# Print the most recent match date
most_recent_date = all_data['Date'].max()

# Calculate the number of days between today and the most recent match date
today = datetime.now().date()
days_difference = (today - most_recent_date.date()).days
print(f"Most recent match date in the dataset: {most_recent_date.date()}")
print(f"This is {days_difference} days out of date"), print()


# Function to generate scores from the row data
def generate_score(row):
    scores = []
    for i in range(1, 6):  # Loop through set numbers 1 to 5
        w_set, l_set = row.get(f'W{i}'), row.get(f'L{i}')
        if pd.notna(w_set) and pd.notna(l_set):
            scores.append(f"{int(w_set)}-{int(l_set)}")
    return ' '.join(scores)

# Apply the function to create the Scores column
all_data['Scores'] = all_data.apply(generate_score, axis=1)

# Save the resulting DataFrame to a CSV file
all_data.to_csv(save_dir / 'all_data.csv', index=False)
print("all_data.csv has been output without errors")

