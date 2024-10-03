# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:46:22 2024   @author: kaalvoetranger@gmail.com

420 line compiler for Tennis matches and ATP tennis players
"""
# %% import dependencies

import os
import warnings
from pathlib import Path
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import re

# Set directories and ignore warnings (edit this sections when deploying)
os.chdir("/Users/kaalvoetranger/Desktop/tennis_app/")
base_dir = os.path.expanduser("~/Desktop/tennis_app/")
data_dir = os.path.join(base_dir, "datasets/")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %% create CompileAndLoad class


class CompileAndLoad:
    def __init__(self, data_dir: str, start_year: int = 2015, end_year: int = 2024):
        """
        Initialize the CompileAndLoad class.

        Parameters:
        - data_dir (str): The directory where the Excel files are located.
        - start_year (int): The starting year for loading data (inclusive).
        - end_year (int): The ending year for loading data (inclusive).
        """
        self.data_dir = Path(data_dir)  # Ensure data_dir is a Path object
        self.start_year = start_year
        self.end_year = end_year
        self.players_out = pd.DataFrame()  # Initialize players_out
        self.all_data = pd.DataFrame()  # Initialize all_data

    def convert_xlsx_to_csv(self, xlsx_file, csv_file):
        """
        Convert an Excel file (.xlsx) to CSV format.

        Parameters:
        xlsx_file (str): The path to the input .xlsx file.
        csv_file (str): The path to the output .csv file.
        """
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel(xlsx_file)

        # Save the DataFrame as a CSV file
        df.to_csv(csv_file, index=False)
        print(f"Converted {xlsx_file} to {csv_file}.")

    def fetch_wikipedia_info(self, player_name):
        """
        Fetch the first paragraph and photo URL of a Wikipedia page for a given tennis player.

        Parameters:
        player_name (str): The name of the tennis player.

        Returns:
        tuple: A tuple containing the first paragraph (str) and the photo URL (str or None).
        """
        try:
            # Wikipedia API for fetching the first paragraph
            api_url = "https://en.wikipedia.org/w/api.php"
            api_params = {"action": "query",
                          "format": "json",
                          "prop": "extracts",
                          "exintro": True,
                          "explaintext": True,
                          "titles": player_name}
            response = requests.get(api_url, params=api_params)
            response.raise_for_status()
            data = response.json()

            # Extract the first paragraph
            page = next(iter(data['query']['pages'].values()))
            first_paragraph = page.get(
                'extract', f"Page for {player_name} not found.").split('\n')[0]

            # Fetch the photo URL from the Wikipedia page
            page_url = f"https://en.wikipedia.org/wiki/{player_name.replace(' ', '_')}"
            response = requests.get(page_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find('table', class_='infobox').find(
                'img') if soup.find('table', class_='infobox') else None
            photo_url = f"https:{img_tag['src']}" if img_tag else None
            print(f"Wiki Info for {player_name} added")
            return first_paragraph, photo_url

        except Exception as e:
            print(f"Error fetching info for {player_name}")
            return f"{e}", None

    def process_player_data(self):
        """
        Process the player data, handle duplicate entries, update Wikipedia info, and calculate player age.
        """
        players = pd.read_csv(self.data_dir / "atp_players_list_v1.csv")

        # Create 'Index Player Name' using initials of first names and full last name
        # players['Index Player Name'] = (players['name_last'] + ' ' +
        #    players['name_first'].apply(lambda x: '.'.join([name[0] for name in
        #                                x.split()]) + '.' if pd.notna(x) else ''
        #                                ))

        # Find duplicate entries based on 'Alt Player Name'
        duplicate_players = players[players.duplicated(
            subset=['Alt Player Name'], keep=False)]

        # Display duplicates
        pd.set_option('display.max_rows', None)
        print(
            "These are the duplicate entries based on 'Player Name' and 'Alt Player Name':")
        print(duplicate_players)

        # Input duplicates to drop from the database
        drop_indices = input(
            "Enter the player_id indices to drop (comma-separated)\n or\nreply ('N') to skip: ")
        if drop_indices == 'N':
            players_out = players
        else:
            # Convert the input string to a list of integers
            drop_indices_list = [int(x.strip())
                                 for x in drop_indices.split(',')]
            # Drop the selected rows from the DataFrame
            players_cleaned = players.drop(index=drop_indices_list)
            print(f"{len(players) - len(players_cleaned)} players removed from DB")
            players_out = players_cleaned

        # Convert the 'dob' column to datetime and calculate player age
        players_out['dob'] = pd.to_datetime(
            players_out['dob'], errors='coerce')
        today = pd.Timestamp(datetime.today().date())
        players_out['age'] = ((today - players_out['dob']).dt.days // 365)

        # Update Wikipedia links and photo URLs if user opts in
        update_wiki = input("Update wiki links?\n'Y' for YES, 'N' for NO):")
        if update_wiki == 'Y':
            players_out[['wikipedia_intro', 'photo_url']] = players_out['Player Name'].apply(
                lambda name: pd.Series(self.fetch_wikipedia_info(name)))

        else:
            print("Skipping Wikipedia updates and moving to matches data")
            time.sleep(2)

        # Reorder DataFrame columns
        first_columns = ['Player Name', 'Alt Player Name', 'dob', 'age',
                         'ioc', 'hand', 'height', 'wikipedia_intro', 'photo_url']
        remaining_columns = [
            col for col in players_out.columns if col not in first_columns]
        self.players_out = players_out[first_columns + remaining_columns]

        # Save the updated DataFrame back to CSV
        players_out.to_csv(
            self.data_dir / 'atp_players_list_v1.csv', index=False)
        print(
            f"Player data updated and saved to {self.data_dir / 'atp_players_list_v1.csv'}")

        # duplicates based on 'Player Name', keeping the row with the most non-null values
        players_compact = players_out.assign(
            non_nulls=lambda x: x.notnull().sum(axis=1)).sort_values(by=['Player Name', 'non_nulls'],
                                                                     ascending=[True, False]).drop_duplicates(subset=['Player Name'],
                                                                                                              keep='first').drop(columns='non_nulls')
        players_compact.to_csv(
            self.data_dir / 'atp_players_v1.csv', index=False)

    def download_and_convert_xlsx(self, year):
        """
        Download the .xlsx file, convert it to .csv, save it, and delete the .xlsx file.

        Parameters:
        year (int): The year for which the file is downloaded and converted.

        Returns:
        DataFrame: The pandas DataFrame containing the data from the CSV.
        """
        # Base URL for downloading .xlsx files
        base_url = 'http://www.tennis-data.co.uk/{year}/{year}.xlsx'

        # Construct the URL and file paths
        url = base_url.format(year=year)
        xlsx_filename = os.path.join(self.data_dir, f'{year}.xlsx')
        csv_filename = os.path.join(self.data_dir, f'{year}.csv')

        # Download the .xlsx file
        print(f"Downloading {xlsx_filename}...")
        response = requests.get(url)
        with open(xlsx_filename, 'wb') as f:
            f.write(response.content)

        # Convert the .xlsx to .csv using pandas
        print(f"Converting {xlsx_filename} to CSV...")
        df = pd.read_excel(xlsx_filename)
        df.to_csv(csv_filename, index=False)

        # Remove the .xlsx file
        print(f"Deleting {xlsx_filename}...")
        os.remove(xlsx_filename)

        # Return the DataFrame for further processing
        return df

    def check_and_update_current_year_csv(self):
        """
        Check the current year's CSV file. If the downloaded file has a more recent date than
        the existing one, replace the existing CSV file; otherwise, warn that no new matches were found.
        """
        current_year = datetime.now().year
        # Define file paths
        csv_filename = os.path.join(self.data_dir, f'{current_year}.csv')

        # Check if the current year's CSV file exists
        if not os.path.isfile(csv_filename):
            print(f"{current_year}.csv not found, downloading...")
            df_downloaded = self.download_and_convert_xlsx(current_year)
            most_recent_date_downloaded = pd.to_datetime(
                df_downloaded['Date']).max()
            print(
                f"Downloaded {current_year}.csv. Most recent match date: {most_recent_date_downloaded}")
        else:
            print(f"{current_year}.csv already exists.")

            # Read the existing CSV to find the most recent match date
            df_existing = pd.read_csv(csv_filename)
            if 'Date' in df_existing.columns:
                df_existing['Date'] = pd.to_datetime(
                    df_existing['Date'], errors='coerce')
                most_recent_date_existing = df_existing['Date'].max()
                print(
                    f"Most recent match date in existing {current_year}.csv: {most_recent_date_existing}")

                # Download the new data
                print(f"Downloading new data for {current_year}...")
                df_downloaded = self.download_and_convert_xlsx(current_year)
                df_downloaded['Date'] = pd.to_datetime(
                    df_downloaded['Date'], errors='coerce')
                most_recent_date_downloaded = df_downloaded['Date'].max()
                print(
                    f"Most recent match date in downloaded data: {most_recent_date_downloaded}")

                # Compare dates
                if most_recent_date_downloaded > most_recent_date_existing:
                    print(f"New data found! Replacing {current_year}.csv.")
                    # Save the new CSV
                    df_downloaded.to_csv(csv_filename, index=False)
                    print(f"{current_year}.csv has been updated.")
                else:
                    print(
                        "No new matches found. The existing file will remain unchanged.")
            else:
                print("Date column not found in existing CSV.")

    def load_matches_data(self) -> pd.DataFrame:
        """
        Load tennis data from csv files in the specified directory 

        Returns:
        - pd.DataFrame: A DataFrame containing the combined data from all csv files.
        """
        # Generate a list of years to include
        # Include the end_year
        years = range(self.start_year, self.end_year + 1)

        # Load CSV files corresponding to the specified years
        dataframes = []

        for year in years:
            file_path = self.data_dir / f'{year}.csv'  # Path object
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Ensure 'Date' column is a datetime object
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Append DataFrame to the list
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"{file_path} does not exist.")

        # Concatenate all DataFrames into a single DataFrame
        if dataframes:
            self.all_data = pd.concat(dataframes, ignore_index=True)
            # Drop rows with missing 'Date' values
            self.all_data.dropna(subset=['Date'], inplace=True)
            print("Matches dataframe saved successfully.")
            print("Have Great Day :) ")

            # Add Scores column
            self.all_data['Scores'] = self.all_data.apply(
                self.generate_score, axis=1)

        else:
            print("No data loaded.")

    # Function to generate Scores
    def generate_score(self, row):
        scores = []
        for i in range(1, 6):  # Loop through set numbers 1 to 5
            w_set, l_set = row.get(f'W{i}'), row.get(f'L{i}')
            if pd.notna(w_set) and pd.notna(l_set):
                scores.append(f"{int(w_set)}-{int(l_set)}")
        return ' '.join(scores)

    def clean_name(self, name):
        """Cleans player names by removing unwanted characters."""
        name = re.sub(r"[.\- ']", "", name)
        name = name.strip().lower()
        return name

    def replace_player_names(self):
        """Replace values in the 'Winner' and 'Loser' columns."""
        if not self.players_out.empty and not self.all_data.empty:
            self.players_out['Clean Player Name'] = self.players_out['Index Player Name'].apply(
                self.clean_name)
            player_mapping = dict(
                zip(self.players_out['Clean Player Name'], self.players_out['Player Name']))

            self.all_data['Clean Winner'] = self.all_data['Winner'].apply(
                self.clean_name)
            self.all_data['Clean Loser'] = self.all_data['Loser'].apply(
                self.clean_name)

            self.all_data['Winner'] = self.all_data['Clean Winner'].replace(
                player_mapping)
            self.all_data['Loser'] = self.all_data['Clean Loser'].replace(
                player_mapping)

            self.all_data.drop(
                columns=['Clean Winner', 'Clean Loser'], inplace=True)
            print("Player names replaced successfully.")
        else:
            print("Player names or match data are not initialized properly.")


"""    
    def replace_player_names(self):
        if not self.players_out.empty and not self.all_data.empty:
            player_mapping = dict(zip(self.players_out['Index Player Name'], self.players_out['Player Name']))
            self.all_data['Winner'] = self.all_data['Winner'].replace(player_mapping)
            self.all_data['Loser'] = self.all_data['Loser'].replace(player_mapping)
            print("Player names replaced successfully.")
        else:
            print("Player names or match data are not initialized properly.")
"""

# %% run the compiler

compiler = CompileAndLoad(data_dir, start_year=2015, end_year=2024)
compiler.process_player_data()
compiler.check_and_update_current_year_csv()
compiler.load_matches_data()
all_data_df = compiler.all_data
players_out = compiler.players_out
compiler.replace_player_names()
all_data_df.to_csv(data_dir + 'matches_v1.csv', index=False)

# %% debugging

# debug compiler
"""
compiler = CompileAndLoad(data_dir, start_year=2015, end_year=2024)
compiler.process_player_data()
compiler.check_and_update_current_year_csv()
compiler.load_matches_data()
# Debug outputs to verify the initialization
print(f"Players DataFrame: {compiler.players_out.head()}")
print(f"All Data DataFrame: {compiler.all_data.head()}")
players_out = compiler.players_out  # or compiler.players_out if it's an attribute
all_data_df = compiler.all_data  # or compiler.all_data if it's an attribute
# Now replace player names
compiler.replace_player_names()
print("Sample of updated dataset:")
print(all_data_df[['Winner', 'Loser']].head(20))  # Adjust the number of rows as needed
"""
# debug bad data capture
"""
# Check for full stops in 'Winner' and only keep specific columns
winner_with_dots = all_data_df[all_data_df['Winner'].str.contains(
    r'\.', na=False)][['Winner', 'Loser', 'Date', 'Scores']]
loser_with_dots = all_data_df[all_data_df['Loser'].str.contains(
    r'\.', na=False)][['Winner', 'Loser', 'Date', 'Scores']]

# Display the results
print("Winners with full stops:")
print(winner_with_dots)

print("\nLosers with full stops:")
print(loser_with_dots)


def has_capitalization(name):
    if pd.isna(name):
        return False  # Skip NaN values
    parts = name.split()
    # Check if each part starts with an uppercase letter
    return all(part[0].isupper() for part in parts if part)


# Check for capitalization issues in 'Winner' and 'Loser' columns
winners_without_capitalization = all_data_df[~all_data_df['Winner'].apply(
    has_capitalization)][['Winner', 'Date']]
losers_without_capitalization = all_data_df[~all_data_df['Loser'].apply(
    has_capitalization)][['Loser', 'Date']]

# Display the results
print("\nLosers without proper capitalization:")
print(losers_without_capitalization)
"""
