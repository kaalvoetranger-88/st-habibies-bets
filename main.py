# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:54:50 2024

@author: kaalvoetranger@gmail.com
---------------------------------
This program has 5 code blocks-->
    1 import dependencies: loads the required libraries and functions
    2 data ingest and initialization: loads datasets and does calculations
    3 layout: 
    
"""

#%% 1 import dependencies:

import os
import io
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from contextlib import redirect_stdout 
import time
import requests

# set local directories 
app_dir = os.getcwd()
data_dir = os.path.expanduser("~/app_dir/datasets/")
os.makedirs(app_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.chdir(app_dir)
os.getcwd()
g_url1 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/matches.csv'
g_url2 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/atp_players.csv'

# import the applications functions  
from elo_funcs import initialize_elo_ratings, get_elo, update_elo, expected_outcome
from elo_funcs import calculate_and_analyze_elo, simulate_match, t_simulate_match, simulate_round, simulate_tournament, player_info_tool
from elo_funcs import decimal_to_fractional, decimal_to_american, fractional_to_decimal, fractional_to_american
from elo_funcs import american_to_decimal, american_to_fractional, calculate_payout, implied_probability

import warnings
# Ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#%% 2 data ingest and initialization


# Load Data
@st.cache_data 
def load_data():
    matches = pd.read_csv(g_url1)
    players = pd.read_csv(g_url2)
    players['Player Name'] = players['name_first'] + ' ' + players['name_last']    
    players['dob'] = players['dob'].astype(str)
    players['dob'] = players['dob'].apply(lambda x: x[:8] if len(x[:8]) == 8 and x[:8].isdigit() else '15000101')
    # Convert to datetime, coercing errors into NaT (Not a Time)
    players['dob'] = pd.to_datetime(players['dob'], format='%Y%m%d', errors='coerce')
    print("players database has been parsed without errors.")
    return matches, players


# matches and players must load before running the elo calcs
matches, players = load_data()


# Cache the main calculation function
@st.cache_data
def get_elo_and_matches():
    elo_df, matches = calculate_and_analyze_elo()
    return elo_df, matches


# Run the calculation once and store results in session state
if 'elo_df' not in st.session_state or 'matches' not in st.session_state:
    st.session_state.elo_df, st.session_state.matches = get_elo_and_matches()

# Access the cached data
elo_df = st.session_state.elo_df
matches = st.session_state.matches

print('initialization success...')


#%% 3 layout

# Sidebar for navigation 
st.sidebar.image('logo.png', width=200)
st.sidebar.title("HaBibie Bets")
tool = st.sidebar.radio("Choose a tool:", ("Player Info", "Player Comparison", "Match Maker", "Odds Converter", "Draw Simulator"))

# Theme Colors
primary_color = "#3498db"
secondary_color = "#2ecc71"
bg_color = "#1e1e1e"
font_color = "#ecf0f1"

# Style overrides  
st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {font_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
    }}
    </style>
    """, unsafe_allow_html=True)


#%% 4 main content and tools


# Mock-up for the main content based on the selected tool
if tool == "Player Info":
    st.header("Player Info Tool:")
    st.write("This tool will display player-specific data.")
    player_info_tool(players, matches, elo_df)
    # Placeholder for player selection and display

elif tool == "Player Comparison":
    st.header("Player Comparison Tool")
    st.write("This tool will allow you to compare two players.")
    # Placeholder for comparison charts

elif tool == "Match Maker":
    st.header("Match Maker Tool")
    st.write("This tool predicts the outcome of a match.")
    # Placeholder for match prediction inputs and output

elif tool == "Odds Converter":
    st.title("Odds Converter")

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Left column: odds converter tool
    with col1:
        # Input for selecting odds type
        odds_type = st.selectbox("Select the type of odds you want to input", ["Decimal", "Fractional", "American"])

        # Odds input section based on selected odds type
        if odds_type == "Decimal":
            decimal_odds = st.number_input("Enter decimal odds:", min_value=1.01, value=2.00)
            fractional_odds = decimal_to_fractional(decimal_odds)
            american_odds = decimal_to_american(decimal_odds)
        elif odds_type == "Fractional":
            fractional_odds = st.text_input("Enter fractional odds (e.g., 5/1):", value="5/1")
            decimal_odds = fractional_to_decimal(fractional_odds)
            american_odds = fractional_to_american(fractional_odds)
        elif odds_type == "American":
            american_odds = st.number_input("Enter American odds (e.g., -200 or 150):", value=150)
            decimal_odds = american_to_decimal(american_odds)
            fractional_odds = american_to_fractional(american_odds)

        # Display the converted odds
        st.subheader("Converted Odds")
        st.write(f"Decimal Odds: {decimal_odds}")
        st.write(f"Fractional Odds: {fractional_odds}")
        st.write(f"American Odds: {american_odds}")

        # Input for wager amount
        wager = st.number_input("Enter your wager:", min_value=1, value=100)

        # Calculate potential payout
        potential_payout = calculate_payout(decimal_odds, wager)

        # Display potential payout
        st.write(f"Potential Payout: ${potential_payout:.2f}")

        # Calculate and display implied probability
        prob = implied_probability(decimal_odds)
        st.write(f"Implied Probability: {prob * 100:.2f}% chance this horse wins")

    # Right column: Markdown text explaining odds and implied probability
    with col2:
        st.markdown("""
    ## Understanding Odds and Probability

    **Decimal Odds:**  
    Decimal odds represent the total payout (including the original stake) for each unit wagered.  
    - Formula: **Payout = Stake x Decimal Odds**  
    - Example: If the decimal odds are 2.50 and you wager $100, your total payout will be $250.

    **Fractional Odds:**  
    Fractional odds represent the profit relative to the stake.  
    - Formula: **Payout = Stake x (Numerator/Denominator)**  
    - Example: If the fractional odds are 5/1, this means you would win $5 for every $1 staked.

    **American Odds:**  
    American odds are either positive or negative and indicate how much you would win on a $100 bet, or how much you need to bet to win $100.  
    - Positive odds: The amount you win on a $100 bet.  
    - Negative odds: The amount you need to bet to win $100.

    ### Implied Probability
    Implied probability represents the likelihood of an outcome occurring as implied by the odds.  
    - Formula: **Implied Probability = 1 / Decimal Odds**

    Example:  
    If the decimal odds are 2.00, the implied probability is **50%** (1 / 2.00 = 0.50).
    """)


# Main section for "Draw Simulator Tool"
elif tool == "Draw Simulator":
    st.header("Draw Simulator Tool")
    st.write("Upload a tournament draw .CSV and simulate results.")

    # Upload CSV for tournament draw
    uploaded_file = st.file_uploader("Upload a CSV with player names", type="csv")

    if uploaded_file is not None:
        # Read player list from uploaded CSV
        player_df = pd.read_csv(uploaded_file)

        # Check if the correct column exists
        if 'Player' in player_df.columns:
            # Display players in the draw with index starting from 1
            player_df.index = player_df.index + 1  # Shift index to start from 1
            st.write("Players in the draw (numbered):")
            st.dataframe(player_df)

            # Allow the user to select which Elo column to use
            available_elo_columns = ['Elo_ALL', 'Elo_Hard', 'Elo_Clay', 'Elo_Grass']  # Example of available columns
            selected_elo_column = st.selectbox("Select Elo Column", available_elo_columns, index=0)

            st.write(f"Selected Elo column: {selected_elo_column}")

            # Simulate tournament button
            if st.button("Simulate Tournament"):
                st.write("Simulating Tournament...")

                match_output = io.StringIO()  # Capture print outputs

                # Simulate tournament and capture printed match details
                with redirect_stdout(match_output):  # Redirect stdout to capture print statements
                    winner = simulate_tournament(player_df['Player'].tolist(), elo_df, elo_column=selected_elo_column)
                    time.sleep(0.1)  # Simulate real-time effect of matches

                # Display tournament winner first
                st.success(f"The tournament winner is: {winner}")

                # Expanded container for match details
                match_details = st.expander("Match Details", expanded=True)
                
                # Display the captured output from the simulation
                with match_details:
                    st.text(match_output.getvalue())

                # Visualize the tournament progress using matplotlib
                fig, ax = plt.subplots()
                ax.barh(player_df['Player'], range(len(player_df), 0, -1))  # Placeholder for actual positions
                ax.set_xlabel("Progress in Tournament")
                ax.set_title("Tournament Simulation Results")
                st.pyplot(fig)

        else:
            st.error("CSV file must contain a column named 'Player' with player names.")           
            
#%% 5 not yet know | scratch

