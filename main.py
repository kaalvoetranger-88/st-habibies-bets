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

# import the application functions from elo_funcs.py  
from elo_funcs import initialize_elo_ratings, get_elo, update_elo, expected_outcome
from elo_funcs import calculate_and_analyze_elo, simulate_match, t_simulate_match, simulate_round, simulate_tournament, player_info_tool
from elo_funcs import decimal_to_fractional, decimal_to_american, fractional_to_decimal, fractional_to_american
from elo_funcs import american_to_decimal, american_to_fractional, calculate_payout, implied_probability
from elo_funcs import calculate_age, plot_player_elo

import warnings
# Ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#%% 2 data ingest, initialization, and caching logic:


# Load match and player data
@st.cache_data 
def load_data():
    matches = pd.read_csv(data_dir + "matches.csv")
    players = pd.read_csv(data_dir + "atp_players.csv")
    players['Player Name'] = players['name_first'] + ' ' + players['name_last']    
    players['dob'] = players['dob'].astype(str)
    players['dob'] = players['dob'].apply(lambda x: x[:8] if len(x[:8]) == 8 and x[:8].isdigit() else '15000101')
    # Convert to datetime, coercing errors into NaT (Not a Time)
    players['dob'] = pd.to_datetime(players['dob'], format='%Y%m%d', errors='coerce')
    print("players database has been parsed without errors.")
    return matches, players


# matches and players must load before running the elo calcs
matches, players = load_data()
max_d = max(matches['tourney_date'])

# Cache the main calculation function
@st.cache_data
def get_elo_and_matches():
    elo_df, matches= calculate_and_analyze_elo()
    print('initialization success...')
    return elo_df, matches


# Run model calculations once and store results in session state
if 'elo_df' not in st.session_state or 'matches' not in st.session_state:
    st.session_state.elo_df, st.session_state.matches = get_elo_and_matches()
    print('Elo dataframe and matches dataframes have been updated.')
# Access the cached data
elo_df = st.session_state.elo_df
matches = st.session_state.matches


# Caching functions to optimize performance
@st.cache_data
def get_player_info(player_input, players, elo_df):
    #st.write("Available index in elo_df:", elo_df.index)                       # Debugging print statements
    #st.write("Available player names in players DataFrame:", players['Player Name'].tolist())  # Debugging print statements

    # Check if the player exists in elo_df index
    player_info = elo_df.loc[elo_df.index.str.contains(player_input, case=False)]

    if not player_info.empty:
        player_name = player_info.index[0]

        # Convert the player names in the players DataFrame to a consistent format
        players_index = players['Player Name'].str.strip().str.lower()
        player_name_lower = player_name.strip().lower()

        if player_name_lower in players_index.str.lower().values:
            # Retrieve the correct index for the player_name
            player_row = players[players['Player Name'].str.strip().str.lower() == player_name_lower]
            dob = player_row['dob'].values[0]  # Retrieve DOB from the players DataFrame
            age = calculate_age(dob)
            #win_percentage = player_row['Win_Percentage'].values[0]
            return {
                "age": age,
                "elo_all": player_info['Elo_ALL'].values[0],
                "elo_grass": player_info['Elo_Grass'].values[0],
                "elo_clay": player_info['Elo_Clay'].values[0],
                "elo_hard": player_info['Elo_Hard'].values[0],
                "win_percentage": f"{player_info['Win_Percentage'].values[0]:.1f}"
            }
            print(f'{player_name} info calculated and cached')
        else:
            st.write(f"Player '{player_name}' not found in the players DataFrame.")
            return None
    else:
        st.write(f"Player '{player_input}' not found in the elo_df DataFrame.")
        return None


@st.cache_data
def prepare_ranking_and_elo_graph(player_name, matches):
    # Ensure that the DataFrame has the necessary columns
    required_columns = ['winner_name', 'loser_name', 'WRank', 'LRank', 'elo_winner_after', 'elo_loser_after', 'tourney_date']
    if all(col in matches.columns for col in required_columns):
        # Filter rows where the player is either a winner or a loser
        player_matches = matches[
            (matches['winner_name'].str.contains(player_name, case=False, na=False)) |
            (matches['loser_name'].str.contains(player_name, case=False, na=False))
        ]
        
        # Create lists to store ranking and Elo rating data
        rankings = []
        elo_ratings = []
        dates = []  # store dates dynamically for plotting
        
        # Determine if player is in the winner or loser column and get the relevant data
        for index, row in player_matches.iterrows():
            if player_name.lower() in row['winner_name'].lower():
                rankings.append(row['WRank'])
                elo_ratings.append(row['elo_winner_after'])
            else:
                rankings.append(row['LRank'])
                elo_ratings.append(row['elo_loser_after'])
            
            dates.append(row['tourney_date'])
        
        # Create a DataFrame for the graph
        graph_data = pd.DataFrame({
            'tourney_date': dates,
            'Player_Rank': rankings,
            'Player_Elo_After': elo_ratings
        })
        print('Ranking and Elo info updated in cache')
        return graph_data
    
    else:
        st.write("One or more required columns are missing in the matches DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame or handle the error appropriately

h2h_record = None
@st.cache_data
def calculate_head_to_head(player_1, player_2, matches):
    # Filter matches where either player 1 or player 2 is involved
    h2h_matches = matches[
        ((matches['winner_name'].str.contains(player_1, case=False)) & 
         (matches['loser_name'].str.contains(player_2, case=False))) |
        ((matches['winner_name'].str.contains(player_2, case=False)) & 
         (matches['loser_name'].str.contains(player_1, case=False)))
    ]
    
    if isinstance(h2h_matches, pd.DataFrame) and not h2h_matches.empty:
        # Count wins for player 1 and player 2
        player_1_wins = len(h2h_matches[h2h_matches['winner_name'].str.contains(player_1, case=False)])
        player_2_wins = len(h2h_matches[h2h_matches['winner_name'].str.contains(player_2, case=False)])
        
        # Handle edge case where there are no losses (to avoid division by zero)
        if player_2_wins > 0:
            h2h_record = player_1_wins / player_2_wins  # Calculate win/loss ratio as a float
        else:
            h2h_record = float(player_1_wins)  # Assign default value if player_2_wins is zero
            
        print(f"H2H info updated in cache: {h2h_record} (as float)")
        
        # Return both the head-to-head record and the match info DataFrame
        return h2h_record, h2h_matches[['tourney_date', 'Tournament', 'surface', 'Round',
                                        'winner_name', 'loser_name', 'Scores']]
    else:
        return 1.0, pd.DataFrame()  # Return default values if no matches are found



def expected_out(player1, player2, surface, matches=matches, weight_surface=0.9, h2h_weight=15):
    weight_all = 1 - weight_surface
    
    # Get Elo ratings from elo_df
    elo1_all = elo_df.loc[player1]
    elo2_all = elo_df.loc[player2]
    
    # Determine surface-specific Elo ratings
    if surface == "Clay":
        elo1_surface = elo1_all[2]
        elo2_surface = elo2_all[2]
    elif surface == "Hard":
        elo1_surface = elo1_all[3]
        elo2_surface = elo2_all[3]
    elif surface == "Grass":
        elo1_surface = elo1_all[1]
        elo2_surface = elo2_all[1]
    else:
        elo1_surface = elo1_all[0]
        elo2_surface = elo2_all[0]
    
    # Combine overall and surface-specific Elo ratings
    combined_elo1 = weight_all * elo1_all[0] + weight_surface * elo1_surface
    combined_elo2 = weight_all * elo2_all[0] + weight_surface * elo2_surface
    
    # Calculate probabilities
    expected_probA = 1 / (1 + 10 ** ((elo2_all[0] - elo1_all[0]) / 400))
    expected_probS = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
    
    # Calculate head-to-head record and get the relevant match info
    h2h_record, h2h_matches_info = calculate_head_to_head(player1, player2, matches)
    # Adjust Elo with head-to-head weight
    combined_elo1 += h2h_weight * h2h_record
    # Calculate head-to-head record and get the relevant match info
    h2h_record, h2h_matches_info = calculate_head_to_head(player2, player1, matches)
    # Adjust Elo with head-to-head weight
    combined_elo2 += h2h_weight * h2h_record
    # Final probability after head-to-head adjustment
    expected_probH = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
    
    # Return probabilities and head-to-head match info DataFrame
    return expected_probA, expected_probS, expected_probH


#%% 3 layout


# Sidebar for navigation 
st.sidebar.code("Build: 0.9        2024-08-27")
st.sidebar.code(f"Most Recent Match = {max_d}")
st.sidebar.divider()
st.sidebar.image('logo.png', width=300)
#st.sidebar.divider()
#st.sidebar.title("HABIBIE BETS")
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


# Main content based on the selected tool
# player info tool
if tool == "Player Info":
    st.header("Player Info Tool:")
    st.write("This tool displays player-specific data.")
    player_info_tool(players, matches, elo_df)
    
# player comparison tool
elif tool == "Player Comparison":
    st.title("Player Comparison Tool")
    # Two columns for player comparison
    col1, col2 = st.columns(2)
    # Player 1 Input
    with col1:
        player_1_input = st.text_input("Search Player 1")
        if player_1_input:
            player_1_info = get_player_info(player_1_input, players, elo_df)
            if player_1_info:
                st.subheader(f"{player_1_input}")
                st.write(f"Age: {player_1_info['age']}")
                #st.write(f"Overall Elo: {player_1_info['elo_all']}")
                #st.write(f"Grass Elo: {player_1_info['elo_grass']}")
                #st.write(f"Clay Elo: {player_1_info['elo_clay']}")
                #st.write(f"Hard Elo: {player_1_info['elo_hard']}")
                st.write(f"Win Percentage: {player_1_info['win_percentage']}%")
                player_1_chart = plot_player_elo(player_1_info, 
                                                 player_1_input,
                                                 position='left')
                st.plotly_chart(player_1_chart)
            else:
                st.write("Player 1 not found.")

    # Player 2 Input
    with col2:
        player_2_input = st.text_input("Search Player 2")
        if player_2_input:
            player_2_info = get_player_info(player_2_input, players, elo_df)
            if player_2_info:
                st.subheader(f"{player_2_input}")
                st.write(f"Age: {player_2_info['age']}")
                #st.write(f"Overall Elo: {player_2_info['elo_all']}")
                #st.write(f"Grass Elo: {player_2_info['elo_grass']}")
                #st.write(f"Clay Elo: {player_2_info['elo_clay']}")
                #st.write(f"Hard Elo: {player_2_info['elo_hard']}")
                st.write(f"Win Percentage: {player_2_info['win_percentage']}%")
                # Display Player 2 Elo Horizontal Bar Chart
                player_2_chart = plot_player_elo(player_2_info, 
                                                 player_2_input,
                                                 position='right')
                st.plotly_chart(player_2_chart)
            else:
                st.write("Player 2 not found.")

    # Plot rankings and Elo ratings over time for both players
    if player_1_input and player_2_input:
        player_1_graph_data = prepare_ranking_and_elo_graph(player_1_input, matches)
        player_2_graph_data = prepare_ranking_and_elo_graph(player_2_input, matches)
        #st.write(f"Player 1 Data:\n{player_1_graph_data.head()}")             # Debugging print statements
        #st.write(f"Player 2 Data:\n{player_2_graph_data.head()}")             # Debugging print statements

        if not player_1_graph_data.empty and not player_2_graph_data.empty:
            fig = go.Figure()

            # Player 1 data
            fig.add_trace(go.Scatter(
                x=player_1_graph_data['tourney_date'],
                y=player_1_graph_data['Player_Rank'],
                mode='lines',
                name=f'{player_1_input} Ranking',
                line=dict(color='blue'),
                yaxis='y1'
                ))

            fig.add_trace(go.Scatter(
                x=player_1_graph_data['tourney_date'],
                y=player_1_graph_data['Player_Elo_After'],
                mode='markers',
                name=f'{player_1_input} Elo Rating',
                line=dict(color='darkblue'),
                yaxis='y2'
                ))

            # Player 2 data
            fig.add_trace(go.Scatter(
                x=player_2_graph_data['tourney_date'],
                y=player_2_graph_data['Player_Rank'],
                mode='lines',
                name=f'{player_2_input} Ranking',
                line=dict(color='red'),
                yaxis='y1'
                ))

            fig.add_trace(go.Scatter(
                x=player_2_graph_data['tourney_date'],
                y=player_2_graph_data['Player_Elo_After'],
                mode='markers',
                name=f'{player_2_input} Elo Rating',
                line=dict(color='darkred'),
                yaxis='y2'
                ))

            # Layout for dual-axis graph
            fig.update_layout(
                title=f"Comparison of {player_1_input} and {player_2_input}",
                xaxis_title="Date",
                yaxis=dict(
                    title='Player Ranking',
                    side='left',
                    showgrid=False,
                    zeroline=False,
                    autorange="reversed",
                    ),
                yaxis2=dict(
                    title='Player Elo Rating',
                    side='right',
                    overlaying='y',
                    showgrid=False,
                    zeroline=False,
                    ),
                legend=dict(x=0.5, y=1.1, xanchor='center', orientation='h'),
                )

            # Display the plot
            st.plotly_chart(fig)
        else:
            st.write("Not enough data to display the comparison graph.")

        # Display Head-to-Head Matches
        st.subheader("Head-to-Head Results")
        h2h_record, h2h_matches = calculate_head_to_head(player_1_input, player_2_input, matches)

        if not h2h_matches.empty:
            st.dataframe(h2h_matches, hide_index=True)
        else:
            st.write(f"No head-to-head matches found between {player_1_input} and {player_2_input}.")
            
            # Create a two-column layout    
        st.divider()
        col3, col4 = st.columns(2)
    
        # Surface selection button in col3
        with col3:
            st.subheader("Estimating Outcomes:")
    
            # Dropdown menu to select the surface
            elo_surface = st.selectbox("Select Court Surface", options=["All Surfaces", "Grass", "Clay", "Hard"])
    
            # Button to calculate the expected probability
            calculate_button = st.button("Calculate Expected Probability")
    
            if calculate_button and player_1_info and player_2_info:
                # Get the Elo ratings based on the chosen surface
                expected_probA, expected_probS, expected_probH = expected_out(player_1_input, player_2_input, elo_surface)
                # Display the result in an f-string
                st.write(f":b: The BASELINE expected probability of {player_1_input} beating {player_2_input} is {expected_probA*100:.1f}%")
                st.write(f":sparkle: The expected probability of {player_1_input} beating {player_2_input} on {elo_surface} is {expected_probS*100:.1f}%")
                st.write(f":eight_spoked_asterisk: The expected probability of {player_1_input} beating {player_2_input} on {elo_surface} with H2H is {expected_probH*100:.1f}%")
                print('Just dropped some dope stats...')
        with col4:
            st.write("placeholder", "")
            st.write("More stuff coming soon")

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

        else:
            st.error("CSV file must contain a column named 'Player' with player names.")           
            
