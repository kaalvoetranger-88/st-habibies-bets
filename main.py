# -*- coding: utf-8 -*-
"""-----------------------------about section---------------------------------.

@author: kaalvoetranger@gmail.com

This script serves as the main execution point for a tennis Elo rating app
that tracks player Elo ratings across different surfaces and provides detailed
match information.
It ingests data, processes match results and Elo calculations, and renders an
interactive dashboard for users to explore statistics and player performance.

The script leverages Streamlit for creating a web-based interface,
while core calculations and functionalities are handled via custom modules.
Users can interact with different tools to view player-specific data,
match histories, and Elo rating changes across different surfaces.

Key functionalities include:
1. Loading and processing player/match datasets
2. Calculating Elo ratings and simulating match results
3. Providing detailed player statistics and head-to-head comparisons
4. Displaying visual insights via charts and tables
5. Explaining underlying mathematical/statistical principles behind Elo ratings

-------------------------------------------------------------------------------

This program has 6 code blocks:

### Code Block 1: Import Dependencies
    - Loads the required libraries such as Streamlit, pandas, numpy, and
      custom modules for Elo calculations.

### Code Block 2: Data Ingest and Initialization
    - Loads the player and match data from CSV files,
      initializes data structures for processing.
    - Performs any necessary preprocessing steps
    - Ensures the datasets are ready for analysis and interaction in the app,
      while handling any potential errors in data loading.

### Code Block 3: Layout Specification
    - Specifies the basic layout and structure of the Streamlit app,
      defining key sections for user interaction.
    - Sets up layout parameters like the sidebar for navigation and main panel
      for viewing data, graphs, and stats.
    - Uses widgets to allow users to interact with different tools

### Code Block 4: Modules
    - Contains all the utility functions required for match outcome predictions
    - Functions include  win/loss probability calculations, betting
      and player comparisons.
    - These functions are called throughout the app to dynamically compute
      values as users interact with the tools.

### Code Block 5: Tools
    - Contains the 6 main tools that users interact with in the dashboard.
    - Each tool offers a different functionality, such as viewing a player’s
      recent match history, comparing two players' Elo ratings over time,
      simulating match and betting outcomes, backtesting etc.
    - These tools are highly interactive, allowing users to visualize
      and analyze data from various perspectives.

### Code Block 6: About Section
    - This block explains the mathematics and statistics behind the system.
    - Provides details on how Elo is calculated, implied probabilities
      and the predictive spaces these reside in.
    - Offers users insights into how the app derives its calculations
      and predictions, making the underlying theory more accessible.

Note:     Ensure that you have the correct file paths and dataset structures
          in place before running the script. The app is modular,
          and most of the heavy computations are done within the functions
          defined in external modules, ensuring scalability and flexibility.


-------------------------------------------------------------------------------
"""
# %% 1 import dependencies

import os
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import logging
import warnings
import requests
from bs4 import BeautifulSoup
from contextlib import redirect_stdout
import time
import io

# set local directories 
app_dir = os.getcwd()
data_dir = os.path.expanduser("~/app_dir/datasets/")
os.makedirs(app_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.chdir(app_dir)
os.getcwd()
g_url1 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/matches_v1.csv'
g_url2 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/atp_players_v1.csv'
c_url = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/construction.jpg'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# import custom Elo and glicko functions
from functions import EloFunctions, GlickoFunctions

# set wide layout
st.set_page_config(layout="wide")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% 2 data loading and caching

@st.cache_data
def load_data():
    try:
        matches = pd.read_csv(g_url1)
        players = pd.read_csv(g_url2)
        players['dob'] = pd.to_datetime(players['dob'], errors='coerce')
        matches['Date'] = pd.to_datetime(matches['Date'])
        logger.info("Data loaded successfully.")
        return matches, players
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        st.error("Error loading data. Check file paths.")
        st.stop()


@st.cache_data
def initialize_data():
    matches, players = load_data()
    max_d = matches['Date'].dt.date.max()
    return matches, players, max_d


# Initialize the data in session state
if 'matches' not in st.session_state or 'players' not in st.session_state:
    st.session_state.matches, st.session_state.players, st.session_state.max_d = initialize_data()


@st.cache_data
def get_elo_and_matches():
    elo_functions = EloFunctions(players=st.session_state.players, matches=st.session_state.matches, K=32)
    elo_df, matches_elo = elo_functions.calculate_and_analyze_elo()
    logger.info("Elo Calculations completed")
    return elo_df, matches_elo


@st.cache_data
def get_glicko_and_matches():
    glicko_functions = GlickoFunctions(players=st.session_state.players, 
                                       matches=st.session_state.matches,
                                       K=12, c=60, decay=0.875)
    glicko_df, matches_glicko = glicko_functions.calculate_and_analyze_glicko(K=12, 
                              H=25, decay=0.85, c=60, DELTA=timedelta(weeks=3))
    logger.info("Glicko Calculations completed")
    return glicko_df, matches_glicko


# Function to handle the rating system toggle
def set_rating_system():
    if st.session_state.rating_system == 'Elo':
        st.session_state.matches = st.session_state.matches_elo
    else:
        st.session_state.matches = st.session_state.matches_glicko


# Cache Elo and Glicko data in session state
if 'elo_df' not in st.session_state or 'matches_elo' not in st.session_state:
    st.session_state.elo_df, st.session_state.matches_elo = get_elo_and_matches()

if 'glicko_df' not in st.session_state or 'matches_glicko' not in st.session_state:
    st.session_state.glicko_df, st.session_state.matches_glicko = get_glicko_and_matches()

# Initialize rating system in session state if not already present
if 'rating_system' not in st.session_state:
    st.session_state.rating_system = 'Elo'  # Default to Elo

# for local debugging uncomment the below lines

#matches, players, max_d = initialize_data()
#elo_functions = EloFunctions(players=players, matches=matches, K=32)
#elo_df, matches_elo = elo_functions.calculate_and_analyze_elo()
#glicko_functions = GlickoFunctions(players=players, matches=matches,
#                                   K=23, c=60, decay=0.875)
#glicko_df, matches_glicko = glicko_functions.calculate_and_analyze_glicko(K=24,
#                            H=25, decay=0.825, c=65, DELTA=timedelta(weeks=2))

players = st.session_state.players
elo_df = st.session_state.elo_df
glicko_df = st.session_state.glicko_df
matches_elo = st.session_state.matches_elo
matches_glicko = st.session_state.matches_glicko
max_d = st.session_state.max_d 
matches = st.session_state.matches
# %% 3 layout

# CSS styling for Custom Header elements
st.markdown(
    """
    <style>
    .custom-header {
        background-color: #ecf83c; /* Custom background color */
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .custom-link {
        color: #A4E8E0;  /* Custom link color */
        text-decoration: none;
        font-weight: normal;
    }
    .custom-link:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True)

# CSS styling for different sections
st.markdown(
    """
    <style>
    .section1 {
        background-color: #f33324; /* Light blue */
        padding: 20px;             /* Padding inside the section */
        border-radius: 10px;       /* Rounded corners */
        margin-bottom: 20px;       /* Space between sections */
    }
    
    .section2 {
        background-color: #4cd7d0; /* Blanched almond */
        padding: 20px;             /* Padding inside the section */
        border-radius: 10px;       /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True)

# HEX color values for App
colors = {
    'primary': '#A4E8E0',
    'secondary': '#ff7f0e',
    'background': '#244843',
    'background2': '#539A64',
    'text': '#FAFAFA',
    'player1': '#a4e8e0',
    'player1a': '#4cd7d0',
    'player2': '#f8ea8c',
    'player2a': '#e1c340',
    'win': '#ecf83c',
    'lose': '#4cd7d0',
    'gauge1': '#4cd7d0',
    'gauge2': '#f8ea8c',
    'gauge3': '#ecf83c',
    'line1': '#ecf83c',
    'line2': '#40b9f8',
    'line1a': '#80b672',
    'line2a': '#044cac'}

# Define header with custom styling
def header():
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        # Toggle switch for Elo or Glicko
        rating_system = st.radio(
            "Rating System", 
            options=['Elo', 'Glicko'], 
            index=0 if st.session_state.rating_system == 'Elo' else 1,
            key='rating_system', horizontal=True)
        
        # Trigger the change in the rating system
        set_rating_system()

    with col2:
        st.markdown(
            '<div class="custom-header">'
            '<a class="custom-link" href="https://www.atptour.com/en/scores/current">ATP This Week</a> | '
            '<a class="custom-link" href="https://www.atptour.com/en/tournaments/shanghai/5014/overview">Rolex Masters Focus</a>'
            '</div>', 
            unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="custom-header">'
            '<a class="custom-link" href="https://www.oddstrader.com/atp/">Odds Trader</a> | '
            '<a class="custom-link" href="https://www.bet.co.za/">BET.co.za</a>'
            '</div>', 
            unsafe_allow_html=True)
    #st.divider()


# App Header
with st.container(border=True):
    header()
#    st.dataframe(st.session_state.matches.iloc[:, [2, 3, 4, 9, 40, 41, 42]].tail(),
#                 hide_index=True)
#    st.write("remove this table before deployment")

# Sidebar for navigation
st.sidebar.code("Build: 1.2*         2024-10-04")
st.sidebar.code(f"Most Recent Match = {max_d}")

st.sidebar.divider()
# App Logo
st.sidebar.image('logo.png', width=300)

# Radio and Buttons to select Tool
st.sidebar.subheader('Choose a tool:')
tool = st.sidebar.radio("Choose a tool:",
                        ("Player Info", "Player Comparison",
                         "Match Maker", "Odds Converter",
                         "Tournament Simulator", "Strategy Simulator"),
                        label_visibility='hidden')
st.sidebar.divider()

# About section that populates below the active tool.
about = st.sidebar.button("🎾 About")


# %% modules


# %% 4 modules

def validate_player_name(elo_df):
    """
    Creates a text input widget for the user to enter a player's name and checks if the name is valid in the `elo_df` DataFrame.

    Args:
        elo_df (pd.DataFrame): DataFrame containing player names as index and Elo ratings.

    Functionality:
        - Displays a text input widget for entering a player's name.
        - Checks if the entered name exists in the index of `elo_df`.
        - Displays a success message if the name is valid, or an error message if it is not.
    """
    # Input for player name
    player_name = st.selectbox("Select a player's name:", 
                               st.session_state.players['Player Name'].unique())
    
    if player_name:
        if player_name in elo_df.index:
            elo_rating = elo_df.loc[player_name, 'Elo_ALL']
            date_dt = pd.to_datetime(elo_df.loc[player_name, 'Most_Recent_Date'])
            date = date_dt.date()
            st.success(f"'{player_name}' is valid. Elo Rating {elo_rating} *calculated on {date}*")
        else:
            st.error(f"'{player_name}' is an :red-background[invalid name] or has :grey-background[no current Elo Rating].")


def gauge_chart(win_percentage, fig_width=350, fig_height=350, title="Win Percentage"):
    """
    Creates and displays a gauge chart representing the win percentage.

    Parameters:
        win_percentage (float): The percentage value to display.
        fig_width (int, optional): The width of the figure. Default is 600.
        fig_height (int, optional): The height of the figure. Default is 400.

    Returns:
        None: Displays the gauge chart using Streamlit's plotly_chart function.
    """
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_percentage,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#f0f6f4'},
            'bgcolor': 'white',
            'steps': [
            {'range': [0, 35], 'color': colors['gauge1']},
            {'range': [36, 65], 'color': colors['gauge2']},
            {'range': [66, 100], 'color': colors['gauge3']}]},
        title={'text': title, 'font': {'size': 32}},
        domain={'x': [0, 1], 'y': [0, 1]}))
    
    # Set figure size
    fig.update_layout(width=fig_width, height=fig_height)
    
    st.plotly_chart(fig)


# depreciated from build 1.2* onwards
def fetch_wikipedia_info(player_name):
    """
    Fetch the first paragraph and photo URL of a Wikipedia page for a given tennis player.

    This function uses the Wikipedia API to get the introductory section and parses
    the page content using BeautifulSoup to retrieve the URL of the main image (if available).

    Parameters:
    player_name (str): The name of the tennis player (e.g., 'Alexander Zverev').

    Returns:
    tuple: A tuple containing:
        - The first paragraph of the player's Wikipedia page if it exists,
        - The URL of the player's photo if it exists,
        - Otherwise, an error message stating the page or photo was not found.
    """
    # Wikipedia API URL
    api_url = "https://en.wikipedia.org/w/api.php"
    
    # Parameters for the API request
    api_params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": player_name
    }


    # Make the API request
    response = requests.get(api_url, params=api_params)
    data = response.json()

    # Extract the first paragraph from the API response
    page = next(iter(data['query']['pages'].values()))
    if 'extract' in page:
        first_paragraph = page['extract'].split('\n')[0]
    else:
        first_paragraph = f"Page for {player_name} not found."

    # Wikipedia page URL for the photo
    page_url = f"https://en.wikipedia.org/wiki/{player_name.replace(' ', '_')}"

    # Make the request to fetch the page content
    response = requests.get(page_url)

    # Check if the request was successful
    if response.status_code != 200:
        return first_paragraph, None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the photo URL
    infobox = soup.find('table', class_='infobox')
    if infobox:
        img_tag = infobox.find('img')
        if img_tag:
            photo_url = 'https:' + img_tag['src']
        else:
            photo_url = None
    else:
        photo_url = None

    return first_paragraph, photo_url


def calculate_win_percentage(player_matches, bin_type, player_name):
    """
    Calculate win percentages based on opponent's rankings or Elo ratings.

    Parameters:
    - player_matches (pd.DataFrame): DataFrame containing match data with columns for winner/loser names and their rankings/ratings.
    - bin_type (str): Type of binning, either 'Ranking' or 'Rating'.
    - player_name (str): Name of the player for whom the win percentages are calculated.

    Returns:
    - pd.DataFrame: DataFrame with win percentages and loss percentages binned by opponent's ranking or rating.
    """
    # Define bins and labels
    if bin_type == 'Ranking':
        bins = [0, 10, 20, 50, 100, float('inf')]
        labels = ['A 0-10', 'B 11-20', 'C 21-50', 'D 51-100', 'E 100+']
        bin_column_winner = 'WRank'
        bin_column_loser = 'LRank'
        opponent_bin_column = 'LRank'  # Opponent's ranking
    elif bin_type == 'Rating' and st.session_state.matches is matches_elo:
        bins = [float('-inf'), 1500, 1599, 1699, 1799, float('inf')]
        labels = ['E <1500', 'D 1500-1599', 'C 1600-1699', 'B 1700-1799', 'A 1800+']
        bin_column_winner = 'elo_winner_before'
        bin_column_loser = 'elo_loser_before'
        opponent_bin_column = 'elo_loser_before'  # Opponent's rating
    elif bin_type == 'Rating' and st.session_state.matches is matches_glicko:
        bins = [float('-inf'), 1500, 1599, 1699, 1799, float('inf')]
        labels = ['E <1500', 'D 1500-1599', 'C 1600-1699', 'B 1700-1799', 'A 1800+']
        bin_column_winner = 'r_winner_before'
        bin_column_loser = 'r_loser_before'
        opponent_bin_column = 'r_loser_before'  # Opponent's rating

    # Bin the data for opponents based on their ranking or rating
    player_matches['Bin'] = pd.cut(
        player_matches.apply(
            lambda row: row[bin_column_loser] if row['Winner'] == player_name else row[bin_column_winner],
            axis=1
        ),
        bins=bins,
        labels=labels,
        right=False
    )
    
    # Calculate win counts and total counts per bin
    win_counts = player_matches[player_matches['Winner'] == player_name].groupby('Bin').size().reindex(labels, fill_value=0)
    total_counts = player_matches.groupby('Bin').size().reindex(labels, fill_value=0)
    
    # Calculate win percentages
    win_percentages = (win_counts / total_counts).fillna(0) * 100

    # Create a DataFrame for plotting and ensure order
    chart_data = pd.DataFrame({
        'Wins': win_percentages,
        'Losses': 100 - win_percentages  # Complement of win percentage
    }, index=labels)  # Set index to ensure bins are ordered
    # Reset index to convert it into a column for x-axis labeling
    chart_data.reset_index(inplace=True)
    chart_data.rename(columns={'index': 'Bin'}, inplace=True)
    return chart_data


@st.cache_data
def get_player_info(player_input, players, elo_df, glicko_df):
    player_info = elo_df.loc[elo_df.index.str.contains(player_input, case=False)].copy()
    player_info2 = glicko_df.loc[glicko_df.index.str.contains(player_input, case=False)].copy()
    if not player_info.empty:
        player_name = player_info.index[0]
        players_index = players['Player Name'].str.strip().str.lower()
        player_name_lower = player_name.strip().lower()

        if player_name_lower in players_index.str.lower().values:
            player_row = players[players['Player Name'].str.strip().str.lower() == player_name_lower]
            return {
                "elo_all": player_info['Elo_ALL'].values[0],
                "elo_grass": player_info['Elo_Grass'].values[0],
                "elo_clay": player_info['Elo_Clay'].values[0],
                "elo_hard": player_info['Elo_Hard'].values[0],
                "win_percentage": f"{player_info['Win_Percentage'].values[0]:.1f}",
                "Ranking": player_info['Most_Recent_Ranking'].values[0],
                "glicko_r": player_info2['R_value'].values[0],
                "glicko_rd": player_info2['RD_value'].values[0],
                "Ranking_Date": player_info['Most_Recent_Date'].dt.date.values[0]                
            }
        else:
            st.error(f"Player '{player_name}' not found in the players DataFrame.")
            return None
    else:
        st.error(f"Player '{player_input}' not found in the elo_df DataFrame.")
        return None


def prepare_ranking_rating_graph(player_name, matches_elo, matches_glicko):
    # Ensure that the DataFrame has the necessary columns
    required_columns_e = ['Winner', 'Loser', 'WRank', 'LRank', 
                          'elo_winner_after', 'elo_loser_after', 'Date']
    required_columns_g = ['Winner', 'Loser', 'WRank', 'LRank', 
                          'r_winner_after', 'r_loser_after', 'Date']
    if all(col in matches_elo.columns for col in required_columns_e):
        # Filter rows where the player is either a winner or a loser
        player_matches_elo = matches_elo[
            (matches_elo['Winner'].str.contains(player_name, case=False, na=False)) |
            (matches_elo['Loser'].str.contains(player_name, case=False, na=False))
        ].copy()
    if all(col in matches_glicko.columns for col in required_columns_g):
        # Filter rows where the player is either a winner or a loser
        player_matches_glicko = matches_glicko[
            (matches_glicko['Winner'].str.contains(player_name, case=False, na=False)) |
            (matches_glicko['Loser'].str.contains(player_name, case=False, na=False))
        ].copy()    
        # Create lists to store ranking and Elo rating data
        rankings = []
        elo_ratings = []
        glicko_ratings = []
        dates = []  # store dates dynamically for plotting
        
        # Determine if player is in the winner or loser column and get the relevant data
        for index, row in player_matches_elo.iterrows():
            if player_name.lower() in row['Winner'].lower():
                rankings.append(row['WRank'])
                elo_ratings.append(row['elo_winner_after'])
            else:
                rankings.append(row['LRank'])
                elo_ratings.append(row['elo_loser_after'])
            
            dates.append(row['Date'])
        for index, row in player_matches_glicko.iterrows():
            if player_name.lower() in row['Winner'].lower():
                glicko_ratings.append(row['r_winner_after'])
            else:
                glicko_ratings.append(row['r_loser_after'])    
        # Create a DataFrame for the graph
        graph_data = pd.DataFrame({
            'Date': dates,
            'Player_Rank': rankings,
            'Player_Elo_After': elo_ratings,
            'Player_Glicko_After': glicko_ratings
        })
        print('Ranking and Rating info updated in cache')
        return graph_data
    
    else:
        st.write("One or more required columns are missing in the matches DataFrames.")
        return pd.DataFrame()  # Return an empty DataFrame or handle the error appropriately


def calculate_head_to_head(player_1, player_2, matches):
    h2h_record = None
    # Filter matches where either player 1 or player 2 is involved
    h2h_matches = matches[
        ((matches['Winner'].str.contains(player_1, case=False)) & 
         (matches['Loser'].str.contains(player_2, case=False))) |
        ((matches['Winner'].str.contains(player_2, case=False)) & 
         (matches['Loser'].str.contains(player_1, case=False)))
    ]
    
    if isinstance(h2h_matches, pd.DataFrame) and not h2h_matches.empty:
        # Count wins for player 1 and player 2
        player_1_wins = len(h2h_matches[h2h_matches['Winner'].str.contains(player_1, case=False)])
        player_2_wins = len(h2h_matches[h2h_matches['Winner'].str.contains(player_2, case=False)])
        
        # Handle edge case where there are no losses (to avoid division by zero)
        if player_2_wins > 0:
            h2h_record = player_1_wins / player_2_wins  # Calculate win/loss ratio as a float
        else:
            h2h_record = float(player_1_wins)  # Assign default value if player_2_wins is zero
            
        logger.info(f"H2H info updated in cache: {h2h_record} (as float)")
        
        # Return both the head-to-head record and the match info DataFrame
        return h2h_record, h2h_matches[['Date', 'Tournament', 'Surface', 'Round',
                                        'Winner', 'Loser', 'Scores']]
    else:
        return 1.0, pd.DataFrame()  # Return default values if no matches are found


def head_to_head_summary(player1, player2, matches):
    # Count wins for each player
    player1_wins = matches[matches['Winner'] == player1].shape[0]
    player2_wins = matches[matches['Winner'] == player2].shape[0]

    # Head-to-head summary
    if player1_wins > player2_wins:
        h2h_summary = f":blue[{player1}] leads the head-to-head :blue-background[{player1_wins}-{player2_wins}]."
    elif player2_wins > player1_wins:
        h2h_summary = f":orange[{player2}] leads the head-to-head :orange-background[{player2_wins}-{player1_wins}]."
    else:
        h2h_summary = f"The head-to-head is tied :grey-background[{player1_wins}-{player2_wins}]."

    # Find most recent match
    recent_match = matches[
        ((matches['Winner'] == player1) & (matches['Loser'] == player2)) | 
        ((matches['Winner'] == player2) & (matches['Loser'] == player1))
    ].sort_values(by='Date', ascending=False).iloc[0]

    recent_winner = recent_match['Winner']
    recent_loser = recent_match['Loser']
    recent_score = recent_match['Scores']
    recent_date = pd.to_datetime(recent_match['Date'], format='%Y%m%d').strftime('%B %d, %Y')

    recent_match_summary = f"The most recent match was won by {recent_winner} with a score of :green-background[{recent_score}] on {recent_date}."

    return h2h_summary, recent_match_summary


def expected_out(player1, player2, Surface, matches=matches, weight_Surface=0.9, h2h_weight=15):
    """
    Calculate the expected outcome probabilities between two players based on Elo ratings, Surface-specific adjustments, and head-to-head record.

    Parameters:
    -----------
    player1 : str
        The name of the first player (e.g., the 'Winner').
    player2 : str
        The name of the second player (e.g., the 'Loser').
    Surface : str
        The Surface on which the match is played. Can be "Clay", "Hard", or "Grass".
    matches : pd.DataFrame, optional
        A DataFrame containing historical match data used to calculate head-to-head (H2H) records (default is `matches`).
    weight_Surface : float, optional
        The weight given to Surface-specific Elo ratings when calculating the combined Elo (default is 0.9). 
        The overall Elo weight is `1 - weight_Surface`.
    h2h_weight : float, optional
        The weight given to the head-to-head (H2H) record in adjusting the final Elo ratings (default is 15).

    Returns:
    --------
    expected_probA : float
        The probability of player1 winning based solely on overall Elo ratings.
    expected_probS : float
        The probability of player1 winning after combining overall and Surface-specific Elo ratings.
    expected_probH : float
        The probability of player1 winning after combining Elo ratings and applying head-to-head adjustments.
    h2h_matches_info : pd.DataFrame
        A DataFrame containing detailed head-to-head match information between the two players.

    Notes:
    ------
    - Elo ratings are combined based on a weighted average of the overall and Surface-specific Elo.
    - Head-to-head records are used to further adjust the combined Elo, enhancing the final win probability.
    - The function considers the specific Surface (Clay, Hard, Grass) for Surface-specific Elo adjustment.
    """
    weight_all = 1 - weight_Surface
    
    # Get Elo ratings from elo_df
    elo1_all = elo_df.loc[player1]
    elo2_all = elo_df.loc[player2]
    
    # Determine Surface-specific Elo ratings
    if Surface == "Clay":
        elo1_Surface = elo1_all[2]
        elo2_Surface = elo2_all[2]
    elif Surface == "Hard":
        elo1_Surface = elo1_all[3]
        elo2_Surface = elo2_all[3]
    elif Surface == "Grass":
        elo1_Surface = elo1_all[1]
        elo2_Surface = elo2_all[1]
    else:
        elo1_Surface = elo1_all[0]
        elo2_Surface = elo2_all[0]
    
    # Combine overall and Surface-specific Elo ratings
    combined_elo1 = weight_all * elo1_all[0] + weight_Surface * elo1_Surface
    combined_elo2 = weight_all * elo2_all[0] + weight_Surface * elo2_Surface
    
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


def plot_player_ratings(player_info, player_name, position='left'):
    """
    Generate a horizontal bar chart displaying the player's Elo ratings across different Surfaces.

    Parameters:
    -----------
    player_info : dict
        A dictionary containing the player's Elo ratings for overall, grass, clay, and hard Surfaces. 
        Keys should include:
        - 'elo_all' for the overall Elo rating,
        - 'elo_grass' for the Elo rating on grass,
        - 'elo_clay' for the Elo rating on clay,
        - 'elo_hard' for the Elo rating on hard.
    player_name : str
        The name of the player whose Elo ratings are being plotted.
    position : str, optional
        The side of the chart the player is on, either 'left' or 'right' (default is 'left'). 
        This determines the alignment and coloring of the bars.

    Returns:
    --------
    fig : plotly.graph_objs._figure.Figure
        A Plotly figure object representing the horizontal bar chart, showing the player's Elo ratings 
        for overall, grass, clay, and hard courts.

    Notes:
    ------
    - The chart is customized based on the player's position, with the 'left' position reversing the x-axis.
    - The x-axis has a default range from 1200 to 2800, representing typical Elo values.
    - Bar colors are customized based on the player's position, with predefined colors for 'player1' and 'player2'.
    """
    # Create a DataFrame from the player_info
    df = pd.DataFrame({
        'Stat': ['RD', 'Glicko', 'Overall Elo', 'Grass Elo', 'Clay Elo', 'Hard Elo'],
        'Value': [
            player_info['glicko_rd'],
            player_info['glicko_r'],
            player_info['elo_all'],
            player_info['elo_grass'],
            player_info['elo_clay'],
            player_info['elo_hard']]
    })

    # Define bar colors and x-axis position based on the player position
    bar_color = colors['player1'] if position == 'left' else colors['player2']
  
    # Create the figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['Stat'],
        x=df['Value'],
        orientation='h',
        marker=dict(color=bar_color),
        name=player_name
    ))

    # Update layout based on player position
    fig.update_layout(
        xaxis=dict(
            title='',
            visible=True,
            range=[1200, 2800],  # Set x-axis range with max of 2000
            autorange='reversed' if position == 'left' else True
        ),
        yaxis=dict(
            title='',
            visible=True if position =='left' else True,
            side='right' if position == 'right' else 'left'
        ),
        showlegend=False,
        width=900,
        height=400)
        
    #fig.update_xaxes(range=[1000, 2500])
    return fig

       
def decimal_to_fractional(decimal_odds):
    numerator = decimal_odds - 1
    denominator = 1
    return f"{int(numerator * denominator)}/{denominator}"


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2:
        american_odds = (decimal_odds - 1) * 100
    else:
        american_odds = -100 / (decimal_odds - 1)
    return int(american_odds)


def fractional_to_decimal(fractional_odds):
    numerator, denominator = map(int, fractional_odds.split('/'))
    return (numerator / denominator) + 1


def fractional_to_american(fractional_odds):
    decimal_odds = fractional_to_decimal(fractional_odds)
    return decimal_to_american(decimal_odds)


def american_to_decimal(american_odds):
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_to_fractional(american_odds):
    decimal_odds = american_to_decimal(american_odds)
    return decimal_to_fractional(decimal_odds)


def calculate_payout(decimal_odds, wager):
    return decimal_odds * wager


def implied_probability(decimal_odds):
    return 1 / decimal_odds


def t_simulate_match(player1, player2, elo_df, elo_column='Elo_ALL'):
    """
    Simulates a match between two players based on their Elo ratings.

    Args:
        player1 (str): Name of the first player.
        player2 (str): Name of the second player.
        elo_df (pd.DataFrame): DataFrame containing Elo ratings for players.
        elo_column (str): The column in the DataFrame containing the Elo ratings to use 
                          (default is 'Elo_ALL').

    Returns:
        str: The name of the winner of the match.
    
    Raises:
        ValueError: If the specified Elo column does not exist in the DataFrame.

    Functionality:
        - Fetches the Elo rating for each player. If a player is not found in the DataFrame, 
          a default Elo rating of 1500 is assigned.
        - Calculates the probability of player 1 winning using the Elo rating formula.
        - Prints the match details, including player names, their Elo ratings, the probability of player 1 winning, 
          and the winner.
        - Returns the name of the winner.
    """
    # Check if the DataFrame contains the necessary columns
    if elo_column not in elo_df.columns:
        raise ValueError(f"Column '{elo_column}' does not exist in the DataFrame")

    # Fetch Elo ratings using the index
    elo1 = elo_df.loc[player1, elo_column] if player1 in elo_df.index else 1500
    elo2 = elo_df.loc[player2, elo_column] if player2 in elo_df.index else 1500
   
    # Calculate probability and determine the winner
    prob_player1_wins = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    winner = player1 if 0.5 <= prob_player1_wins else player2

    print(f"Match: {player1} (Elo: {elo1}) vs {player2} (Elo: {elo2})")
    print(f"Probability {player1} wins: {prob_player1_wins:.2f}")
    print(f"Winner: {winner}")
    print('.')

    return winner


def simulate_round(draw, elo_df, elo_column='Elo_ALL'):
    """
    Simulates a round of matches for the given tournament draw.

    Args:
        draw (list): List of player names participating in the current round.
        elo_df (pd.DataFrame): DataFrame containing Elo ratings for players.
        elo_column (str): The column in the DataFrame containing the Elo ratings to use 
                          (default is 'Elo_ALL').

    Returns:
        list: List of winners from the current round.

    Functionality:
        - Pairs players from the `draw` in the order they appear.
        - Simulates a match for each pair using `t_simulate_match` and appends the winner to the `winners` list.
        - Returns the list of winners who will advance to the next round.
    """
    winners = []
    for i in range(0, len(draw), 2):
        player1 = draw[i]
        player2 = draw[i + 1]
        winner = t_simulate_match(player1, player2, elo_df, elo_column)
        winners.append(winner)
        # time.sleep(0.1) for debugging 
    return winners


def simulate_tournament(player_list, elo_df, elo_column='Elo_ALL'):
    """
    Simulates an entire tournament using the given list of players and their Elo ratings.

    Args:
        player_list (list): List of player names participating in the tournament.
        elo_df (pd.DataFrame): DataFrame containing Elo ratings for players.
        elo_column (str): The column in the DataFrame containing the Elo ratings to use 
                          (default is 'Elo_ALL').

    Returns:
        str: The name of the tournament winner.

    Functionality:
        - Simulates multiple rounds, starting with the list of players in `player_list`.
        - Calls `simulate_round` to determine the winners for each round.
        - Continues simulating rounds until only one player remains, who is declared the winner.
        - Prints the winner and details of each round.
        - Returns the name of the tournament winner.
    """
    round_num = 1
    current_round = player_list
    
    while len(current_round) > 1:
        print(f"\nSimulating Round {round_num}")
        winners = simulate_round(current_round, elo_df, elo_column)
        current_round = winners
        round_num += 1
    
    # Final winner
    print(f"\nTournament Winner: {current_round[0]}")
    return current_round[0]


def calculate_ev(expected_prob, odds, stake):
    # Calculate the implied probability from the bookmaker's odds
    implied_prob = 1 / odds
    
    # Calculate potential profit (odds - 1) * stake
    profit = (odds - 1) * stake
    
    # Calculate EV
    ev = (expected_prob * profit) - ((1 - expected_prob) * stake)
    
    return ev


def calculate_vig(implied_prob_1, implied_prob_2):
    total_implied_prob = implied_prob_1 + implied_prob_2
    vig = total_implied_prob - 1
    return vig


def live_match_maker(player_1, player_2, Surface, model, odds_1, odds_2, stake=100):
    """
    Calculate the Expected Value (EV) for betting on a tennis match between two players,
    using different Elo models, and provide a decision on whether to place a bet.

    Parameters:
    -----------
    player_1: str Name of Player 1.
    player_2: str Name of Player 2.
    Surface: str The Surface type (Hard, Clay, Grass)
    model: int The model to use for probability calculation:
                0 - Overall Elo, 1 - Surface-specific Elo, 2 - Head-to-Head Elo
    odds_1 : float The bookmaker's odds for Player 1. 
    odds_2 : float The bookmaker's odds for Player 2.
    stake : float, optional the amount staked on the bet. Default is R100

    Returns:
    --------
    None
        Prints the Expected Value (EV) for both players and a recommendation
        on whether betting on each player has a positive or negative expected value.
    """
    
    # Calculate the expected probability of Player 1 winning
    prob_1 = expected_out(player_1, player_2, Surface=Surface)[model]
    
    # Player 2's probability is the complement of Player 1's probability
    prob_2 = 1 - prob_1
    
    # Calculate EV for Player 1 and Player 2
    ev_1 = calculate_ev(prob_1, odds_1, stake)
    ev_2 = calculate_ev(prob_2, odds_2, stake)
    
    # Display the results
    print(f"Expected Value for betting on {player_1}: R{ev_1:.2f}")
    print(f"Expected Value for betting on {player_2}: R{ev_2:.2f}")
    
    # Decision-making
    if ev_1 > 0:
        print(f"Betting on {player_1} has a positive expected value.")
    else:
        print(f"Betting on {player_1} has a negative expected value.")
    
    if ev_2 > 0:
        print(f"Betting on {player_2} has a positive expected value.")
    else:
        print(f"Betting on {player_2} has a negative expected value.")
    
    return ev_1, ev_2


def select_odds(row, odds_type='best'):
    """
    Selects the appropriate odds for a given match.
    
    Args:
    - row (pd.Series): A row from the matches DataFrame.
    - odds_type (str): 'B365' for Bet365 odds, 'PS' for Pinnacle odds, or 'best' for the best of both.
    
    Returns:
    - Tuple (float, float): (Odds for Player 1, Odds for Player 2)
    """
    if odds_type == 'B365':
        return row['B365W'], row['B365L']
    elif odds_type == 'PS':
        return row['PSW'], row['PSL']
    elif odds_type == 'best':
        odds_1 = max(row['B365W'], row['PSW'])
        odds_2 = max(row['B365L'], row['PSL'])
        return odds_1, odds_2
    else:
        raise ValueError("Invalid odds_type. Choose 'B365', 'PS', or 'best'.")


def calculate_custom_stake(ev, prob, odds):
    """
    Calculate the stake dynamically based on EV, probability, and odds.
    """
    base_stake = 100
    stake = base_stake #* (1 + ev / 100) * prob
    
    # Adjust for high odds
    if odds > 5.0:
        stake = stake * 0.60
    elif odds > 2.0:
        stake = stake * 0.80
    return stake


def calculate_bet(row, prob_column, odds_type='best', criterion='EV', stake=100, 
                 min_ev=0, min_odds=1.5, max_odds=3.0, min_prob=0.55, 
                 Surface=None, custom_stake=True):
    """
    Calculates the bet decision, EV, implied probability, and additional filters for a match.
    
    Args:
    - row (pd.Series): A row from the matches DataFrame.
    - prob_column (str): The column name for the expected probability (e.g., 'expected_probA').
    - odds_type (str): 'B365', 'PS', or 'best' for selecting odds.
    - criterion (str): 'EV' to bet based on Expected Value, or 'implied' to bet based on implied probability.
    - stake (int): The amount to bet per match.
    - min_ev (float): Minimum EV to place a bet.
    - min_odds (float): Minimum odds to consider a bet.
    - max_odds (float): Maximum odds to consider a bet.
    - min_prob (float): Minimum expected probability for placing a bet.
    - Surface (str): Optional filter to place bets only on specific Surfaces.
    - custom_stake (bool): Whether to use a custom stake calculation.
    
    Returns:
    - dict: Contains bet decision, EV, implied probability, filters, and P&L.
    """
    # Select the appropriate odds
    odds_1, odds_2 = select_odds(row, odds_type=odds_type)
    
    # Calculate implied probabilities from the odds
    implied_prob_1 = 1 / odds_1
    implied_prob_2 = 1 / odds_2
    
    # Get the expected probability for Player 1 and Player 2
    expected_prob_1 = row[prob_column]
    expected_prob_2 = 1 - expected_prob_1
    
    # Calculate EV for both players
    ev_1 = stake * (expected_prob_1 * odds_1 - 1)
    ev_2 = stake * (expected_prob_2 * odds_2 - 1)
    
    # Filter based on Surface
    if Surface and row['Surface'] != Surface:
        return {'bet_on': None, 'ev_1': ev_1, 'ev_2': ev_2, 'profit_loss': 0, 
                'vigorish': None, 'stake': 0}
    
    # Filter based on EV and odds range
    if ev_1 < min_ev and ev_2 < min_ev:
        return {'bet_on': None, 'ev_1': ev_1, 'ev_2': ev_2, 'profit_loss': 0, 
                'vigorish': None, 'stake': 0}
    if not (min_odds <= odds_1 <= max_odds) or not (min_odds <= odds_2 <= max_odds):
        return {'bet_on': None, 'ev_1': ev_1, 'ev_2': ev_2, 'profit_loss': 0, 
                'vigorish': None, 'stake': 0}
    
    # Determine the bet decision based on the criterion and apply stake
    if criterion == 'EV':
        if ev_1 > ev_2 and ev_1 > 0:
            bet_on = row['Winner']
            selected_ev = ev_1
            if custom_stake:
                stake = calculate_custom_stake(selected_ev, expected_prob_1, odds_1)
                profit_loss = stake * (odds_1 - 1) if row['Winner'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
            else:
                profit_loss = stake * (odds_1 - 1) if row['Winner'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
        elif ev_2 > ev_1 and ev_2 > 0:
            bet_on = row['Loser']
            selected_ev = ev_2
            if custom_stake:
                stake = calculate_custom_stake(selected_ev, expected_prob_2, odds_2)                
                profit_loss = stake * (odds_2 - 1) if row['Loser'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
            else:
                profit_loss = stake * (odds_2 - 1) if row['Loser'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
        else:
            bet_on = None
            profit_loss = 0
            selected_ev = 0
    elif criterion == 'implied':
        if implied_prob_1 < expected_prob_1 and ev_1 > 0:
            bet_on = row['Winner']
            selected_ev = ev_1
            if custom_stake:
                stake = calculate_custom_stake(selected_ev, expected_prob_1, odds_1)
                profit_loss = stake * (odds_1 - 1) if row['Winner'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
            else:
                profit_loss = stake * (odds_1 - 1) if row['Winner'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
        elif implied_prob_2 < expected_prob_2 and ev_2 > 0:
            bet_on = row['Loser']
            selected_ev = ev_2
            if custom_stake:
                stake = calculate_custom_stake(selected_ev, expected_prob_2, odds_2) 
                profit_loss = stake * (odds_2 - 1) if row['Loser'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
            else:
                profit_loss = stake * (odds_2 - 1) if row['Loser'] == row['Winner'] else -stake
                print(f"Bet R{stake:.0f} on {bet_on}")
        else:
            bet_on = None
            profit_loss = 0
            selected_ev = 0
    else:
        raise ValueError("Invalid criterion. Choose 'EV' or 'implied'.")
    
    # Calculate the vigorish
    vigorish = (1 / implied_prob_1 + 1 / implied_prob_2 - 1) * 100
    
    # Return a summary of the bet decision and results
    return {
        'bet_on': bet_on,
        'ev_1': ev_1,
        'ev_2': ev_2,
        'implied_prob_1': implied_prob_1,
        'implied_prob_2': implied_prob_2,
        'profit_loss': profit_loss,
        'vigorish': vigorish,
        'stake': stake
    }


def backtest_strategy(matches, prob_column, odds_type='B365', criterion='implied', 
                      stake=100, min_ev=5, min_odds=1.5, max_odds=2.5, min_prob=0.6, 
                      Surface=None, custom_stake=True):
    """
    Backtests the betting strategy over the given matches and adds the results to the matches DataFrame.
    
    Args:
    - matches (pd.DataFrame): DataFrame containing match data.
    - prob_column (str): The column name for the expected probability (e.g., 'expected_probA').
    - odds_type (str): 'B365', 'PS', or 'best' for selecting odds.
    - criterion (str): 'EV' to bet based on Expected Value, or 'implied' to bet based on implied probability.
    - stake (int): The amount to bet per match.
    
    Returns:
    - dict: Contains overall performance metrics and the modified matches DataFrame with betting results.
    """
    # Create a new DataFrame for bets results
    matches_bet = matches.copy()
    
    # Initialize new columns to store results
    matches_bet['bet_on'] = None
    matches_bet['profit_loss'] = 0.0
    matches_bet['vigorish'] = 0.0
    matches_bet['ev_1'] = 0.0
    matches_bet['ev_2'] = 0.0
    matches_bet['implied_prob_1'] = 0.0
    matches_bet['implied_prob_2'] = 0.0
    matches_bet['odds_type'] = odds_type
    matches_bet['criterion'] = criterion
    matches_bet['stake'] = 0
    
    total_profit = 0
    total_bets = 0
    wins = 0
    losses = 0
    
    # Applying the calculate_bet function within a loop
    for idx, match in matches_bet.iterrows():
        bet_result = calculate_bet(match, prob_column=prob_column, odds_type=odds_type, 
                                criterion=criterion, min_ev=min_ev, min_odds=min_odds, 
                                max_odds=max_odds, min_prob=min_prob, 
                                Surface=Surface, custom_stake=custom_stake)

        # Update profit and loss
        if bet_result['bet_on']:
            total_bets += 1
            total_profit += bet_result['profit_loss']
            if bet_result['profit_loss'] > 0:
                wins += 1
            else:
                losses += 1
            
            # Store the result in the DataFrame
            matches_bet.at[idx, 'bet_on'] = bet_result['bet_on']
            matches_bet.at[idx, 'profit_loss'] = bet_result['profit_loss']
            matches_bet.at[idx, 'vigorish'] = bet_result['vigorish']
            matches_bet.at[idx, 'ev_1'] = bet_result['ev_1']
            matches_bet.at[idx, 'ev_2'] = bet_result['ev_2']
            matches_bet.at[idx, 'implied_prob_1'] = bet_result['implied_prob_1']
            matches_bet.at[idx, 'implied_prob_2'] = bet_result['implied_prob_2']
            matches_bet.at[idx, 'stake'] = bet_result['stake']
            matches_bet['selected_ev'] = matches_bet.apply(
            lambda row: row['ev_1'] if row['Winner'] == row['bet_on'] else row['ev_2'], axis=1)
    
    matches_bet['bet_eval'] = matches_bet['profit_loss'].apply(
    lambda x: 1 if x > 0 else (0 if x < 0 else -99))
    
    matches_bet['wonTEST'] = matches_bet['profit_loss'] > 0
    
    # Calculate final metrics
    win_rate = wins / total_bets if total_bets > 0 else 0
    total_stake = sum(matches_bet['stake'])
    roi = (total_profit / (total_bets * stake)) * 100 if total_bets > 0 else 0
    
    matches_bet['Date'] = pd.to_datetime(matches_bet['Date'])     
    # Return a summary of the backtest results and the updated matches DataFrame
    return {
        'total_profit': total_profit,
        'total_stake': total_stake,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'roi': roi,
        'matches_bet': matches_bet
    }


def analyze_backtest_results(backtest_results):
    total_profit = backtest_results['total_profit']
    total_bets = backtest_results['total_bets']
    win_rate = backtest_results['win_rate']
    roi = backtest_results['roi']
    
    print(f"Total Profit: R{total_profit:.2f}")
    print(f"Total Bets: {total_bets}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"ROI: {roi:.2f}%")
    
    # Additional analysis
    bets_with_positive_ev = backtest_results[backtest_results['EV'] > 0]
    print(f"Bets with Positive EV: {len(bets_with_positive_ev)}")
    
    # Check if certain criteria or odds types are performing better
    performance_by_odds_type = backtest_results.groupby('odds_type')['total_profit'].sum()
    print("Total Profit by Odds Type:")
    print(performance_by_odds_type)
    
    performance_by_criterion = backtest_results.groupby('criterion')['total_profit'].sum()
    print("Total Profit by Criterion:")
    print(performance_by_criterion)


def plot_win_rate_by_ev(backtest_results):
    """
    Plots the win rate by expected value (EV) using backtest results.

    This function groups the backtest results into EV bins and calculates
    the win rate for each bin. The win rate is defined as the percentage
    of profitable outcomes (profit_loss > 0) in each EV bin.

    Args:
        backtest_results (pd.DataFrame): A DataFrame containing the backtest results,
                                         with columns 'selected_ev' (expected value)
                                         and 'profit_loss' (the outcome of each trade).
    """
    # Filter matches that were bet on
    backtest_results1 = backtest_results[backtest_results['bet_eval'] >= 0]
    
    # Calculate win (1 if profit_loss > 0, else 0) for each result
    backtest_results1['won'] = backtest_results1['profit_loss'] > 0
    
    # Create EV bins
    ev_bins = pd.cut(backtest_results['selected_ev'], bins=10)
    
    # Calculate win rate for each EV bin
    win_rate_by_ev = backtest_results1.groupby(ev_bins)['won'].mean()

    # Create bar plot using Plotly
    fig = go.Figure(data=[
        go.Bar(x=win_rate_by_ev.index.astype(str), 
               y=win_rate_by_ev.values,
               marker_color=colors['win'])  # Change bar color here
    ])

    # Update layout for the figure
    fig.update_layout(
        title="Win Rate by Expected Value (EV)",
        xaxis_title="Expected Value (EV)",
        yaxis_title="Win Rate",
        xaxis=dict(tickangle=-45),
        template="plotly_dark"  # Dark theme for Plotly
    )

    # Display the plot inside Streamlit
    st.plotly_chart(fig)


def plot_win_rate_by_prob(backtest_results):
    # Filter matches that were bet on
    backtest_results = backtest_results[backtest_results['bet_eval'] >= 0]
    
    backtest_results['won'] = backtest_results['profit_loss'] > 0
    prob_bins = pd.cut(backtest_results['expected_probA'], bins=10)
    win_rate_by_prob = backtest_results.groupby(prob_bins)['won'].mean()

    fig = go.Figure(data=[
        go.Bar(x=win_rate_by_prob.index.astype(str),
               y=win_rate_by_prob.values,
               marker_color=colors['win'])
    ])

    fig.update_layout(
        title="Win Rate by Probability",
        xaxis_title="Calculated Probability (Baseline Model)",
        yaxis_title="Win Rate",
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )

    # Use st.plotly_chart to display the figure inside Streamlit
    st.plotly_chart(fig)


def plot_roi_by_ev(backtest_results):
    backtest_results = backtest_results[backtest_results['bet_eval'] >= 0]
    backtest_results['EV_bins'] = pd.cut(backtest_results['selected_ev'], bins=10)
    roi_by_ev = backtest_results.groupby('EV_bins')['profit_loss'].sum() / 100  # R100 stake

    fig = go.Figure(data=[
        go.Bar(x=roi_by_ev.index.astype(str), y=roi_by_ev.values,
               marker_color = colors['win'])
    ])

    fig.update_layout(
        title="ROI by EV",
        xaxis_title="EV Range",
        yaxis_title="ROI (%)",
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )

    # Use st.plotly_chart to display the figure inside Streamlit
    st.plotly_chart(fig)


def plot_profit_by_Surface(backtest_results):
    profit_by_Surface = backtest_results.groupby('Surface')['profit_loss'].sum()

    fig = go.Figure(data=[
        go.Bar(x=profit_by_Surface.index, y=profit_by_Surface.values,
               marker_color=colors['win'])
    ])

    fig.update_layout(
        title="Profit by Surface",
        xaxis_title="Surface",
        yaxis_title="Profit (Rands)",
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )

    # Use st.plotly_chart to display the figure inside Streamlit
    st.plotly_chart(fig)


def plot_profit_by_round(backtest_results):
    backtest_results = backtest_results[backtest_results['bet_eval'] >= 0]
    profit_by_round = backtest_results.groupby('Round')['profit_loss'].sum()

    fig = go.Figure(data=[
        go.Bar(x=profit_by_round.index, y=profit_by_round.values,
               marker_color=colors['win'])
    ])

    fig.update_layout(
        title="Profit by Tournament Round",
        xaxis_title="Round",
        yaxis_title="Profit (Rands)",
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )

    # Use st.plotly_chart to display the figure inside Streamlit
    st.plotly_chart(fig)


def ev_vs_actual_profit(backtest_results):
    backtest_results = backtest_results[backtest_results['bet_eval'] >= 0]
    # Define color conditions: green for positive profit, red for negative profit
    colors = ['yellow' if profit > 0 else 'lightblue' for profit in backtest_results['profit_loss']]

    fig = go.Figure(data=[
        go.Scatter(
            x=backtest_results['selected_ev'], 
            y=backtest_results['profit_loss'],
            mode='markers',
            marker=dict(
                color=colors  # Apply the conditional color list
            )
        )
    ])

    fig.update_layout(
        title="EV vs Actual Profit",
        xaxis_title="Expected Value (EV)",
        yaxis_title="Total Profit",
        shapes=[dict(type='line', 
                     x0=min(backtest_results['selected_ev']), 
                     x1=max(backtest_results['selected_ev']),
                     y0=0, y1=0, line=dict(color='red', dash='dash'))],  # Profit/Loss threshold
        template="plotly_dark"
    )

    # Use st.plotly_chart to display the figure inside Streamlit
    st.plotly_chart(fig)


def player_info_tool(players, matches, elo_df, glicko_df):
    # Search bar to search for a player by name
    leftcol, rightcol = st.columns([1,2])
    with leftcol:
        player_name_input = st.selectbox("Select a player's name:", 
                                         st.session_state.players['Player Name'],
                                         placeholder="Enter Player Name Here",
                                         label_visibility='hidden',
                                         index=None)
    st.divider()
    if player_name_input:
       
        # Filter the players DataFrame by player name
        filtered_players = players[players['Player Name'].str.contains(player_name_input, case=False)]

        if not filtered_players.empty:
            for _, row in filtered_players.iterrows():
                
                # Display player details from the players and matches DataFrame
                st.subheader(f"Player: {row['Player Name']}")
                logger.info(f"Player data for {row['Player Name']} Generated")
                with st.expander(":green[Biography]", expanded=True):
                    # Use columns to split the layout into Photo and blurb
                    col1, col2 = st.columns([2,5])
                
                    # Left column for player photo
                    with col1:
                        # Now using photo_url directly from the players DataFrame
                        if pd.notna(row['photo_url']):
                            st.image(row['photo_url'], 
                                     caption="Image from Wikipedia")
                        else:
                            st.error("No photo available")
                    # Right column for player info
                    with col2:
                        st.write(f":green[Nationality:] {row['ioc']} ")
                        if pd.notna(row['age']):
                            st.write(f":green[Age:] {row['age']:.0f} years")   
                        else:
                            st.error("Not in Database")
                        if pd.notna(row['height']):
                            st.write(f":green[Height:] {row['height']:.0f}cm")   
                        else:
                            st.error("Not in Database")  
                        if row['hand'] == 'R':
                            st.write(':green[Plays:] Right Handed')
                        elif row['hand'] == 'L':
                            st.write('Plays: Left Handed')
                        elif row['hand'] == 'A':
                            st.write('Plays: Like a Ninja')
                        else:
                            st.write("Plays unknown")

                        # Elo information
                        player_info = get_player_info(player_name_input, players, elo_df, glicko_df)
                        if player_name_input in elo_df.index:
                            st.write(f":green[Elo Rating:] {player_info['elo_all']:.0f}")
                        # Glicko information (R_value and RD_value)
                        if player_name_input in glicko_df.index:
                            st.write(f":green[Glicko Rating:] {glicko_df.at[player_name_input, 'R_value']:.0f} *(RD of {glicko_df.at[player_name_input, 'RD_value']:.0f})*")
                    
                                    
                        st.write(f":green[ATP Ranking:] {player_info['Ranking']:.0f} *({player_info['Ranking_Date']})* ")
                        if abs((max_d - player_info['Ranking_Date']).days) <= 30:
                                st.write(':green-background[Active]')
                        elif abs((max_d - player_info['Ranking_Date']).days) <= 90:
                                st.write(':blue-background[> 30 days since last match]')
                        elif abs((max_d - player_info['Ranking_Date']).days) <= 300:
                                st.write(':grey-background[Inactive]')
                        else:
                                st.write(':red-background[Retired]')
                                           
                        # Player biography directly from the players DataFrame
                        st.write(row['wikipedia_intro'])
                        
                with st.expander(":green[Statistics]", expanded=False):
                    # Use columns to split the sliders
                    col1, col2 = st.columns([3, 5])   
                    with col1:
                        with st.container(height=120, border=True):
                            # Surface filter selection
                            Surface = st.radio("Surface", options=['All', 'Grass', 'Clay', 'Hard'], 
                                index=0, horizontal=True, key='s1')
                    with col2:
                        with st.container(height=120, border=True):
                            # Date range filter selection
                            d_range = st.select_slider(label="Time",
                                                       options=['Career', 'Year', 'Recent'],
                                    label_visibility='hidden')

                            # Filter player matches by inputs:
                            player_matches = matches[
                            (matches['Winner'] == row['Player Name']) |
                            (matches['Loser'] == row['Player Name'])
                            ].copy()
                    # Apply Surface filter
                    if Surface != 'All':
                        player_matches = player_matches[player_matches['Surface'] == Surface]
                    # Apply date range filter
                    if d_range == 'Year':
                        current_year = datetime.now().year
                        player_matches = player_matches[
                            player_matches['Date'].dt.year == current_year]
                    elif d_range == 'Recent':
                            player_matches = player_matches.tail(30)                  
                    # build stats from filtered data
                    total_matches = len(player_matches)
                    wins = len(player_matches[player_matches['Winner'] == row['Player Name']])
                    win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0

                    col1, col2 = st.columns([3, 5])
                    # display match and win percentage
                    with col1:
                        st.markdown(f"<h1 style='font-size:32px; text-align: center;'>Matches Played</h1>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='font-size:96px; text-align: center;'>{total_matches}</h1>", 
                                    unsafe_allow_html=True)
                        gauge_chart(win_percentage)
                    # display win percentage by rank
                    with col2:
                        with st.container():
                            st.write("")
                            title_holder = st.empty()
                            bin_type = st.radio("Select Binning Type", 
                                                options=['Ranking', 'Rating'],
                                                label_visibility='hidden',
                                                horizontal=True)

                            # Calculate win percentages
                            chart_data = calculate_win_percentage(player_matches,
                                                                  bin_type,
                                                                  player_name_input)
    
                            # Create the bar chart
                            title_holder.subheader(f'Win Percentage by Opponent {bin_type}')
                            st.bar_chart(chart_data, x='Bin',
                                         #title=f'Win Percentage by Opponent {bin_type}',
                                         x_label=f'Opponent {bin_type}',
                                         color=[colors['lose'],colors['win']],
                                         use_container_width=True)  
                    player_matches['Date'] = player_matches['Date'].dt.date
                    selected_columns = ['Date', 'Surface', 'Tournament',
                                        'Round', 'Winner', 'Loser', 'Scores']
                    player_matches_d = player_matches[selected_columns].copy()
                    # Display the filtered data
                    st.dataframe(player_matches_d, hide_index=True,
                                 use_container_width=True)
                    
                with st.expander(":green[Ranking and Ratings]", expanded=False):               
                    # Show recent match results from `matches`
                    elo_matches = matches_elo[
                        (matches_elo['Winner'] == row['Player Name']) | 
                        (matches_elo['Loser'] == row['Player Name'])
                        ].copy()                    
                    glicko_matches = matches_glicko[
                        (matches_glicko['Winner'] == row['Player Name']) | 
                        (matches_glicko['Loser'] == row['Player Name'])
                        ].copy()
                    # Track player ranking and Elo ratings
                    elo_matches['Player_Rank'] = elo_matches.apply(
                        lambda x: x['WRank'] if x['Winner'] == row['Player Name'] else x['LRank'], axis=1)
                
                    elo_matches['Player_Elo_After'] = elo_matches.apply(
                        lambda x: x['elo_winner_after'] if x['Winner'] == row['Player Name'] else x['elo_loser_after'], axis=1)
                    
                    glicko_matches['Player_Glicko_After'] = glicko_matches.apply(
                        lambda x: x['r_winner_after'] if x['Winner'] == row['Player Name'] else x['r_loser_after'], axis=1)
                
                    # Sort the matches by date for the plot
                    elo_matches['Date'] = pd.to_datetime(elo_matches['Date'])
                    elo_matches = elo_matches.sort_values(by='Date')
                
                    if not elo_matches.empty:
                    # Plot rankings and Elo rating over time using Plotly
                        fig = go.Figure()

                        # Add the ranking data to the plot
                        fig.add_trace(go.Scatter(
                            x=elo_matches['Date'],
                            y=elo_matches['Player_Rank'],
                            mode='lines+markers',
                            name='ATP Ranking',
                            line=dict(color=colors['line1']),
                            yaxis='y1'))
                        # Add the Elo/Glicko rating data to the plot
                        fig.add_trace(go.Scatter(
                            x=elo_matches['Date'],
                            y=elo_matches['Player_Elo_After'],
                            mode='lines+markers',
                            name='Elo Rating',
                            line=dict(color=colors['line1a']),
                            yaxis='y2'))
                        fig.add_trace(go.Scatter(
                            x=elo_matches['Date'],
                            y=glicko_matches['Player_Glicko_After'],
                            mode='lines+markers',
                            name='Glicko Rating',
                            line=dict(color=colors['gauge2']),
                            yaxis='y2'))
                        # Layout for dual axis
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis=dict(
                                title='ATP Ranking',
                                side='left',
                                showgrid=False,
                                zeroline=False,
                                autorange="reversed",  # Rankings are better when they are lower
                                ),
                            yaxis2=dict(
                                title=' Elo & Glicko Rating',
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
                        st.write("No recent matches found.")
                    
                with st.expander(":red[Betting Information]", expanded=False):
                    # Use columns to split the layout into Photo and blurb
                    col1, col2 = st.columns([2,5])
                    st.write(":red[*This section will be available in next build*]")
                    st.image(c_url, width=300)
        

        else:
            st.write("No player found with that name.")
    else:
        st.write("Enter a player name to search.")


def player_comparison_tool(players, elo_df, matches):
    # Two columns for player comparison
    leftcol, rightcol = st.columns(2)
    # Player 1 Input
    with leftcol:
        player_1_input = st.selectbox(f"Search :blue[Player 1]", 
                                         st.session_state.players['Player Name'],
                                         placeholder="Enter Player Name Here", index=None)    
    # Player 2 Input
    with rightcol:
        player_2_input = st.selectbox(f"Search :orange[Player 2]", 
                                          st.session_state.players['Player Name'],
                                          placeholder="Enter Player Name here", index=None)    
    st.divider()  
      
    # Set conditions for comparison
    if player_1_input:
        player_1_info = get_player_info(player_1_input, players, elo_df, glicko_df)
        # Filter the players DataFrame by player name
        filtered_players1 = players[players['Player Name'].str.contains(player_1_input, case=False, na="")]
        
        if player_2_input:
            # Ensure 'Player Name' column has no NaN values for filtering
            player_2_info = get_player_info(player_2_input, players, elo_df, glicko_df)
            # Filter the players DataFrame by player name
            filtered_players2 = players[players['Player Name'].str.contains(player_2_input, case=False, na="")]            
        
        else: 
            st.error("Need 2 recently active players to Compare")
            return
            
        if not filtered_players1.empty and not filtered_players2.empty:
            # General section for comparison of both players
            with st.expander(label=':green[Biography Comparison]', expanded=False):       
      
                player1col, player2col = st.columns(2)
                
                with player1col:
                    with st.container(border=True): 
                        st.markdown(f"""
                                    <p style='text-align: center; font-size: 32px;'>
                                    <span style='color: {colors['player1']}; font-weight: bold;'>{player_1_input}</span>
                                    </p>
                                    """, unsafe_allow_html=True)
                        with st.container(height=400, border=False):
                            # Now using photo_url directly from the players DataFrame
                            if filtered_players1['photo_url'].notna().all():
                                st.image(filtered_players1['photo_url'].values[0], 
                                         caption="Image from Wikipedia")
                            else:
                                st.error("No photo available")
                            # Player 1 stats
                        if player_1_input in elo_df.index:
                        
                            if abs((max_d - player_1_info['Ranking_Date']).days) <= 30:
                                st.markdown("<p style='text-align: right; font-size: 18px;'>"
                                            "<span style='background-color: #21c354; color: white;'>Active</span></p>", unsafe_allow_html=True)
                            elif abs((max_d - player_1_info['Ranking_Date']).days) <= 90:
                                st.markdown("<p style='text-align: right; font-size: 18px;'>"
                                            "<span style='background-color: blue; color: white;'>> 30 days since last match</span></p>", unsafe_allow_html=True)
                            elif abs((max_d - player_1_info['Ranking_Date']).days) <= 300:
                                st.markdown("<p style='text-align: right; font-size: 18px;'>"
                                            "<span style='background-color: grey; color: white;'>Inactive</span></p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p style='text-align: right; font-size: 18px;'>"
                                    "<span style='background-color: red; color: white;'>Retired</span></p>", unsafe_allow_html=True)                       
                            st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Nationality: </span> "
                                        f"<span style='color: white;'> {filtered_players1['ioc'].iloc[0]} ", 
                                        unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Age: </span> "
                                        f"<span style='color: white;'> {filtered_players1['age'].iloc[0]:.0f} years ", 
                                        unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Height: </span> "
                                        f"<span style='color: white;'> {filtered_players1['height'].iloc[0]:.0f}cm ", 
                                        unsafe_allow_html=True)
                            if filtered_players1['hand'].iloc[0] == 'R':
                                st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Right Handed", 
                                            unsafe_allow_html=True)
                            elif filtered_players1['hand'].iloc[0] == 'L':
                                st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Left Handed ", 
                                            unsafe_allow_html=True)
                            elif filtered_players1['hand'].iloc[0] == 'A':
                                st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Like a Ninja", 
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Unknown", 
                                            unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: right; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>ATP Ranking: </span> "
                                        f"<span style='color: white;'>{player_1_info['Ranking']:.0f} "
                                        f"<span style='font-style: italic;'>( {player_1_info['Ranking_Date']} )</span></span></p>", 
                                        unsafe_allow_html=True)
                            player_1_chart = plot_player_ratings(player_1_info, player_1_input, position='left')
                            st.plotly_chart(player_1_chart)
                            #st.write(f"{filtered_players1['wikipedia_intro']}")
                                                
                with player2col:
                    with st.container(border=True):
                        st.markdown(f"""
                                    <p style='text-align: center; font-size: 32px;'>
                                    <span style='color: {colors['player2']}; font-weight: bold;'>{player_2_input}</span>
                                    </p
                                    """, unsafe_allow_html=True)
                        with st.container(height=400, border=False):
                            if filtered_players2['photo_url'].notna().all():
                                st.image(filtered_players2['photo_url'].values[0], 
                                         caption="Image from Wikipedia")
                            else:
                                st.error("No photo available")
                        # player 2 stats
                        if player_2_input in elo_df.index:
                        
                            if abs((max_d - player_2_info['Ranking_Date']).days) <= 30:
                                st.markdown("<p style='text-align: left; font-size: 18px;'>"
                                            "<span style='background-color: #21c354; color: white;'>Active</span></p>", unsafe_allow_html=True)
                            elif abs((max_d - player_2_info['Ranking_Date']).days) <= 90:
                                st.markdown("<p style='text-align: left; font-size: 18px;'>"
                                            "<span style='background-color: blue; color: white;'>> 30 days since last match</span></p>", unsafe_allow_html=True)
                            elif abs((max_d - player_2_info['Ranking_Date']).days) <= 300:
                                st.markdown("<p style='text-align: left; font-size: 18px;'>"
                                            "<span style='background-color: grey; color: white;'>Inactive</span></p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p style='text-align: left; font-size: 18px;'>"
                                            "<span style='background-color: red; color: white;'>Retired</span></p>", unsafe_allow_html=True)

                            st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Nationality: </span> "
                                        f"<span style='color: white;'> {filtered_players2['ioc'].iloc[0]} ", 
                                        unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Age: </span> "
                                        f"<span style='color: white;'> {filtered_players2['age'].iloc[0]:.0f} years ", 
                                        unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>Height: </span> "
                                        f"<span style='color: white;'> {filtered_players2['height'].iloc[0]:.0f}cm ", 
                                        unsafe_allow_html=True)
                            if filtered_players2['hand'].iloc[0] == 'R':
                                st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Right Handed", 
                                            unsafe_allow_html=True)
                            elif filtered_players2['hand'].iloc[0] == 'L':
                                st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Left Handed ", 
                                            unsafe_allow_html=True)
                            elif filtered_players2['hand'].iloc[0] == 'A':
                                st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Like a Ninja", 
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                            f"<span style='color: #21c354;'>Plays: </span> "
                                            f"<span style='color: white;'> Unknown", 
                                            unsafe_allow_html=True)                        
                            st.markdown(f"<p style='text-align: left; font-size: 18px;'>"
                                        f"<span style='color: #21c354;'>ATP Ranking: </span> "
                                        f"<span style='color: white;'>{player_2_info['Ranking']:.0f} "
                                        f"<span style='font-style: italic;'>( {player_2_info['Ranking_Date']} )</span></span></p>", 
                                        unsafe_allow_html=True)
                            player_2_chart = plot_player_ratings(player_2_info, player_2_input, position='right')
                            st.plotly_chart(player_2_chart)
                            #st.write(filtered_players2['wikipedia_intro'])

            # Show basic statistics
            with st.expander(":green[Statistics Comparison]", expanded=False):
                    # Use columns to split the sliders
                col1, col2 = st.columns([2,5])
                with col1:
                    player_selector = st.selectbox(label="Choose a Player",
                                                       options=[f'{player_1_input}',
                                                                f'{player_2_input}'])
                    
                col1, col2 = st.columns([3, 5])   
                with col1:
                    with st.container(height=120, border=True):
                        # Surface filter selection
                        Surface = st.radio("Surface", options=['All', 'Grass', 'Clay', 'Hard'], 
                                index=0, horizontal=True, key='s1')
                with col2:
                    with st.container(height=120, border=True):
                        # Date range filter selection
                        d_range = st.select_slider(label="Time",
                                                       options=['Career', 'Year', 'Recent'],
                                    label_visibility='hidden')
                        
                        # choose matches based on player selected
                        if player_selector == f'{player_1_input}':
                            player_matches = matches[
                                (matches['Winner'] == player_1_input) |
                                (matches['Loser'] == player_1_input)
                                ].copy()
                        else:
                            player_matches = matches[
                                (matches['Winner'] == player_2_input) |
                                (matches['Loser'] == player_2_input)
                                ].copy()
                            
                        # Apply Surface filter
                        if Surface != 'All':
                            player_matches = player_matches[player_matches['Surface'] == Surface]
                        # Apply date range filter
                        if d_range == 'Year':
                            current_year = datetime.now().year
                            player_matches = player_matches[
                                player_matches['Date'].dt.year == current_year]
                        elif d_range == 'Recent':
                            player_matches = player_matches.tail(30)                  
                        # build stats from filtered data
                        total_matches = len(player_matches)
                        if player_selector == f'{player_1_input}':
                            wins = len(player_matches[player_matches['Winner'] == player_1_input])
                        else:
                            wins = len(player_matches[player_matches['Winner'] == player_2_input])
                        
                        win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0

                col1, col2 = st.columns([3, 5])
                # display match and win percentage
                with col1:
                    st.markdown(f"<h1 style='font-size:32px; text-align: center;'>Matches Played</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='font-size:96px; text-align: center;'>{total_matches}</h1>", 
                                unsafe_allow_html=True)
                    gauge_chart(win_percentage)
                    # display win percentage by rank
                with col2:
                    with st.container():
                        st.write("")
                        title_holder = st.empty()
                        bin_type = st.radio("Select Binning Type", 
                                             options=['Ranking', 'Rating'],
                                             label_visibility='hidden',
                                             horizontal=True)

                        # Display win percentages
                        if player_selector == f'{player_1_input}':
                            chart_data = calculate_win_percentage(player_matches,
                                                              bin_type,
                                                              player_1_input)
                        else:
                            chart_data = calculate_win_percentage(player_matches,
                                                              bin_type,
                                                              player_2_input)
    
                        # Create the bar chart
                        title_holder.subheader(f'Win Percentage by Opponent {bin_type}')
                        st.bar_chart(chart_data, x='Bin',
                                         #title=f'Win Percentage by Opponent {bin_type}',
                                         x_label=f'Opponent {bin_type}',
                                         color=[colors['lose'],colors['win']],
                                         use_container_width=True)  
                        player_matches['Date'] = player_matches['Date'].dt.date
                        selected_columns = ['Date', 'Surface', 'Tournament',
                                        'Round', 'Winner', 'Loser', 'Scores']
                        player_matches_d = player_matches[selected_columns].copy()
                # Display the filtered data
                if player_selector == f'{player_1_input}':
                    st.write(f'You can view details on all {total_matches} matches for :blue[{player_1_input}] below:')
                else:
                    st.write(f'You can view details on all {total_matches} matches for :orange[{player_2_input}] below:')
                
                st.dataframe(player_matches_d, hide_index=True,
                                     use_container_width=True)
            
            # Plot rankings and Elo ratings over time for both players
            with st.expander(label=':green[ATP Ranking and Elo Ratings Comparison]', expanded=False):
    
                player_1_graph_data = prepare_ranking_rating_graph(player_1_input, matches_elo, matches_glicko)
                player_2_graph_data = prepare_ranking_rating_graph(player_2_input, matches_elo, matches_glicko)

                if not player_1_graph_data.empty and not player_2_graph_data.empty:
                    fig = go.Figure()
                    
                    # Player 1 data
                    fig.add_trace(go.Scatter(
                        x=player_1_graph_data['Date'],
                        y=player_1_graph_data['Player_Rank'],
                        mode='lines',
                        name=f'{player_1_input} Ranking',
                        line=dict(color=colors['player1']),
                        yaxis='y1'
                        ))
                    fig.add_trace(go.Scatter(
                        x=player_1_graph_data['Date'],
                        y=player_1_graph_data['Player_Elo_After'],
                        mode='markers',
                        name=f'{player_1_input} Elo Rating',
                        line=dict(color=colors['player1a']),
                        yaxis='y2'
                        ))
                    fig.add_trace(go.Scatter(
                        x=player_1_graph_data['Date'],
                        y=player_1_graph_data['Player_Glicko_After'],
                        mode='markers',
                        name=f'{player_1_input} Glicko Rating',
                        line=dict(color=colors['line2a']),
                        yaxis='y2'
                        ))
                    # Player 2 data
                    fig.add_trace(go.Scatter(
                        x=player_2_graph_data['Date'],
                        y=player_2_graph_data['Player_Rank'],
                        mode='lines',
                        name=f'{player_2_input} Ranking',
                        line=dict(color=colors['player2']),
                        yaxis='y1'
                        ))
                    fig.add_trace(go.Scatter(
                        x=player_2_graph_data['Date'],
                        y=player_2_graph_data['Player_Elo_After'],
                        mode='markers',
                        name=f'{player_2_input} Elo Rating',
                        line=dict(color=colors['player2a']),
                        yaxis='y2'
                        ))
                    fig.add_trace(go.Scatter(
                        x=player_2_graph_data['Date'],
                        y=player_2_graph_data['Player_Glicko_After'],
                        mode='markers',
                        name=f'{player_2_input} Glicko Rating',
                        line=dict(color=colors['line1']),
                        yaxis='y2'
                        ))
                    # Layout for dual-axis graph
                    fig.update_layout(xaxis_title="Date",
                        yaxis=dict(
                            title='Player Ranking',
                            side='left',
                            showgrid=False,
                            zeroline=False,
                            autorange="reversed",
                            ),
                        yaxis2=dict(
                            title='Player Elo/Glicko Rating',
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
                    st.error("Not enough data to display the comparison graph.")
                
            # Head to Head info for both players
            with st.expander(label=':green[Head-to-Head]', expanded=True):   
                # Display Head-to-Head Matches
                h2h_record, h2h_matches = calculate_head_to_head(player_1_input, player_2_input, matches)
                if not h2h_matches.empty:
                    h2h_matches['Date'] = h2h_matches['Date'].dt.date
                    # Reorder columns with 'Date' as the first column and drop 'Date'
                    columns_order = ['Date'] + [col for col in h2h_matches.columns if col != 'Date' and col != 'Date']
                    h2h_matches_display = h2h_matches[columns_order]
                    sum1, sum2 = head_to_head_summary(player_1_input, player_2_input, h2h_matches)
                    st.write(sum1)
                    st.write(sum2)
                    st.dataframe(h2h_matches_display, hide_index=True, use_container_width=True)
                else:
                    st.write(f"No head-to-head matches found between {player_1_input} and {player_2_input}.")                
            
            with st.expander(label=':green[Expected Outcome]', expanded=False):   
                leftcol, rightcol = st.columns([1,2])
                with leftcol:
                    with st.container(border=True):
                        # Menu to select the Surface
                        Surface_C = st.radio("Match Surface", options=['All', 'Grass', 'Clay', 'Hard'], 
                                           index=0, horizontal=True, key='s7')
                        s_weight = st.slider('Surface Weight (%)', min_value=0.5,
                                             max_value=1.0, value=0.8, key='w2')
                        h_weight = st.slider('H2H Elo Boost (Points)', min_value=0,
                                             max_value=50, value=25, key='w5',
                                             step=5)
                        model_select = st.selectbox("Gauge",options=["🅱️ BASELINE",
                                                                         "❇️ Surface",
                                                                         "✳️ Head-to-Head"])
                with rightcol:
                    col1, col2, col3 = st.columns([1,10,1])
                    with col2:
                        # Get the Elo ratings based on the chosen Surface
                        expected_probA, expected_probS, expected_probH = expected_out(player_1_input, player_2_input,
                                                                                      Surface_C, matches.copy(),
                                                                                      weight_Surface = s_weight,
                                                                                      h2h_weight=h_weight)
                        
                        if model_select == "🅱️ BASELINE":   
                            gauge_chart(expected_probA*100, 450, 350,
                                        title=f"Probability {player_1_input} wins")
                        elif model_select == "❇️ Surface":   
                            gauge_chart(expected_probS*100, 450, 350, title=f"Probability {player_1_input} wins")
                        else:
                            gauge_chart(expected_probH*100, 450, 350, title=f"Probability {player_1_input} wins")
                        # Display the result in an f-string       
                st.write(f":b: The BASELINE expected probability of :blue[{player_1_input}] beating {player_2_input} is {expected_probA*100:.1f}%")
                st.write(f":sparkle: The expected probability of :blue[{player_1_input}] beating {player_2_input} on {Surface_C} is {expected_probS*100:.1f}%")
                st.write(f":eight_spoked_asterisk: The expected probability of :blue[{player_1_input}] beating {player_2_input} on {Surface_C} with H2H is {expected_probH*100:.1f}%")

    else:
        st.write("Enter :blue[Player1] and :orange[Player2] to search and compare")


def match_maker_tool():
    # Inputs for the Match Maker Tool
    st.divider()
    leftcol, rightcol = st.columns(2)
    
    with leftcol:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                player_1 = st.selectbox(f"Enter :blue[Player 1]", 
                                                 st.session_state.players['Player Name'], index=None) 
                player_2 = st.selectbox(f"Enter :orange[Player 2]", 
                                                 st.session_state.players['Player Name'], index=None)
            with col2:
                odds_1 = st.number_input("Bookies odds for Player 1", min_value=1.01, value=1.96, step=0.01)
                odds_2 = st.number_input("Bookies odds for Player 2", min_value=1.01, value=1.94, step=0.01)
              
            Surface = st.select_slider("Select the court Surface", ["Hard", "Clay", "Grass"])
            model = st.radio("Select the **Elo Model** to use", 
                             [("Overall", 0), 
                              ("Surface-Specific", 1), 
                              ("Head-to-Head", 2)], 
                             format_func=lambda x: x[0])
            stake = st.number_input("Enter the stake amount in Rands", min_value=1, value=100, step=1)
            calc_button = st.button("Calculate")
            
    with rightcol:
        # Button to calculate the outcome
        if calc_button == True:
            with st.spinner('Calculating Expected Outcome and Expected Value...'): 
                if player_1 and player_2:
                                                  
                    # Calculate the expected probability of Player 1 winning using the selected model
                    prob_1 = expected_out(player_1, player_2, Surface=Surface)[model[1]]
            
                    if odds_1 and odds_2:
                        # Calculate the implied probabilities from the bookmaker's odds
                        implied_prob_1 = 1 / odds_1
                        implied_prob_2 = 1 / odds_2
                        vig = calculate_vig(implied_prob_1, implied_prob_2)

                        # Run the Match Maker Tool to calculate EV
                        ev_1, ev_2 = live_match_maker(player_1, player_2, Surface, model[1], odds_1, odds_2, stake)
                                              
                        # Display Vigor
                        st.write(f"**Bookmaker's Vigor:** {vig:.1%}")            
                        # Display decision-making
                        if ev_1 > 0:
                            photo_url = players.loc[players['Player Name'] == player_1, 'photo_url'].dropna().values[0]
                            st.image(photo_url, caption=f"Image of {player_1}") 
                            st.success(f"Betting on {player_1} has a positive EV")
                            # Display the results
                            st.write(f"**{player_1} EV:** R{ev_1:.2f}")
                            st.write(f"Our Model gives them a {prob_1:.1%} chance to win while the bookies say {implied_prob_1:.1%}")
                            st.write(f"**{player_2} EV:** R{ev_2:.2f}")
                            st.write(f"Our Model gives them a {1 - prob_1:.1%} chance to win while the bookies say {implied_prob_2:.1%}")
                            st.error(f"Betting on {player_2} has a negative EV")
                        elif ev_2 > 0:
                            photo_url = players.loc[players['Player Name'] == player_2, 'photo_url'].dropna().values[0]
                            st.image(photo_url, caption=f"Image of {player_2}") 
                            st.success(f"Betting on {player_2} has a positive EV")
                            # Display the results
                            st.write(f"**{player_2} EV:** R{ev_2:.2f}")
                            st.write(f"Our Model gives them a {1-prob_1:.1%} chance to win while the bookies say {implied_prob_2:.1%}")
                            st.write(f"**{player_1} EV:** R{ev_1:.2f}")
                            st.write(f"Our Model gives them a {prob_1:.1%} chance to win while the bookies say {implied_prob_1:.1%}")
                            st.error(f"Betting on {player_1} has a negative EV")
                        else:
                            st.error(f"Vig is too high on this one")
                    else:
                        st.warning("Please enter the bookmakers odds for each player")
                else:
                    st.warning("Please enter both player names.")


def odds_tool():
    import streamlit as st

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Left column: odds converter tool
    with col1:
        with st.container(border=True):
            # Input for selecting odds type
            odds_type = st.selectbox("Select the :green-background[type] of odds you want to input",
                                     ["Decimal", "Fractional", "American"],
                                     label_visibility='visible')

            # Odds input section based on selected odds type
            if odds_type == "Decimal":
                decimal_odds = st.number_input("Enter :green-background[decimal] odds:", min_value=1.01, value=2.00)
                fractional_odds = decimal_to_fractional(decimal_odds)
                american_odds = decimal_to_american(decimal_odds)
            elif odds_type == "Fractional":
                fractional_odds = st.text_input("Enter :green-background[fractional] odds (e.g., 5/1):", value="5/1")
                decimal_odds = fractional_to_decimal(fractional_odds)
                american_odds = fractional_to_american(fractional_odds)
            elif odds_type == "American":
                american_odds = st.number_input("Enter :green-background[american] odds (e.g., -200 or 150):", value=150)
                decimal_odds = american_to_decimal(american_odds)
                fractional_odds = american_to_fractional(american_odds)
            
            #st.divider()
            # Display the converted odds
            st.header(":green[Converted Odds]")
            st.subheader(f":green-background[Decimal Odds:]    {decimal_odds}")
            st.subheader(f":green-background[Fractional Odds:] {fractional_odds}")
            st.subheader(f":green-background[American Odds:]   {american_odds}")

            # Input for wager amount
            wager = st.number_input("Enter your wager (R):", min_value=1, value=100)

            # Calculate potential payout
            potential_payout = calculate_payout(decimal_odds, wager)

            # Display potential payout
            st.subheader(f":green-background[Potential Payout:] R{potential_payout:.2f}")

            # Calculate and display implied probability
            prob = implied_probability(decimal_odds)
            st.subheader(f":green-background[Implied Probability:]")
            st.subheader(f"**{prob * 100:.1f}%** chance this horse wins")

    # Right column: Markdown text explaining odds and implied probability
    with col2:
        with st.expander(f":orange-background[Understand Odds and Probability]" ):
            #st.header(":green[Understand Odds and Probability]")
            st.markdown("""
                    <style>
                    .subheading {
                        color: #4CAF50;  
                        font-size: 24px;
                        font-weight: bold;
                        margin-top: 20px;
                        }
                    .example {
                        margin-left: 15px;
                        font-style: italic;
                        color: #F88F48;
                        }
                    .formula {
                        background-color: #06402B;
                        padding: 8px;
                        border-radius: 5px;
                        font-weight: bold;
                        }
                    </style>
                    
        <div class="subheading">Decimal Odds:</div>
        Decimal odds represent the total payout (including the original stake) for each unit wagered.  
        <div class="formula">Formula: Payout = Stake x Decimal Odds</div>
        <div class="example">Example: If the decimal odds are 2.50 and you wager R100, your total payout will be R250.</div>

        <div class="subheading">Fractional Odds:</div>
        Fractional odds represent the profit relative to the stake.  
        <div class="formula">Formula: Payout = Stake x (Numerator/Denominator)</div>
        <div class="example">Example: If the fractional odds are 5/1, this means you would win R5 for every R1 staked.</div>

        <div class="subheading">American Odds:</div>
        American odds are either positive or negative and indicate how much you would win on a R100 bet, or how much you need to bet to win R100.  
        <div class="formula">Positive odds: The amount you win on a R100 bet.</div>  
        <div class="formula">Negative odds: The amount you need to bet to win R100.</div>

        <div class="subheading">Implied Probability:</div>
        Implied probability represents the likelihood of an outcome occurring as implied by the odds.  
        <div class="formula">Formula: Implied Probability = 1 / Decimal Odds</div>
        
        <div class="example">Example: If the decimal odds are 2.00, the implied probability is 50% (1 / 2.00 = 0.50).</div>
        """, unsafe_allow_html=True)

 
def tournament_draw_simulator_tool(elo_df):
    """
    Upload a CSV file containing player names and simulate a tournament draw 
    based on Elo ratings.

    Args:
        elo_df (pd.DataFrame): A DataFrame containing player Elo ratings 
                               with columns for different Surfaces (e.g., 'Elo_ALL', 'Elo_Hard', etc.).

    Functionality:
        - Allows the user to upload a CSV file with a list of players.
        - Displays the uploaded player list, starting the index from 1.
        - Allows the user to select an Elo column for simulating the tournament.
        - Simulates the tournament using the selected Elo ratings and outputs the match details.
        - Displays the winner and an expandable section with detailed match results.
    """
 # Create two-column layout
    col1, col2 = st.columns([2,1])

    # Left column: Tournament draw simulator functionality
    with col1:
        # Upload CSV for tournament draw
        uploaded_file = st.file_uploader("Upload a CSV with player names", type="csv")

        if uploaded_file is not None:
            # Read player list from uploaded CSV
            player_df = pd.read_csv(uploaded_file)

            # Check if the correct column exists
            if 'Player' in player_df.columns:
                # Display players in the draw with index starting from 1
                player_df.index = player_df.index + 1  # Shift index to start from 1
                with st.expander(":green[Players in the draw:]"):
                    st.dataframe(player_df)

                # Allow the user to select which Elo column to use
                available_elo_columns = ['Elo_ALL', 'Elo_Hard', 'Elo_Clay', 'Elo_Grass']  # Example of available columns
                selected_elo_column = st.radio("Select Elo Model",
                                               available_elo_columns,
                                               horizontal=True, index=0)

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

    # Right column: Placeholder for additional content
    with col2:
        with st.container(border=True):
            st.markdown("## Quick Check")
            validate_player_name(elo_df)


def backtest_strategy_tool(matches, max_d, backtest_strategy):
    # container to display inputs
    with st.expander("Strategy Dashboard", expanded=True):
        st.write("Tweak your strategy using these inputs")    
        # Create a three-column layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # Parameter Selection
            st.subheader("Strategy Parameters")

            # Allow user to select the probability column, odds type, and criterion
            prob_column = st.selectbox("Probability Model",
                                       options=['expected_probA', 'expected_probS', 'expected_probH'], index=0)
            criterion = st.selectbox("Decision Metric", options=['EV', 'implied'], index=0)
            odds_type = st.selectbox("Betting Odds Source", options=['B365', 'PS', 'best'], index=1)
            Surface = st.selectbox("Surface (Optional)", options=['All', 'Grass', 'Clay', 'Hard'], index=0)

        with col2:
            # User inputs for other parameters
            st.subheader("Betting Parameters")
            stake = st.number_input("Stake", min_value=1, max_value=10000, value=100)
            min_ev = st.slider("Minimum EV", min_value=-stake, max_value=stake, value=0)
            oddsmin, oddsmax = st.columns(2)
            with oddsmin:
                min_odds = st.number_input("Minimum Odds", min_value=1.0, max_value=10.0, value=1.5)
            with oddsmax:
                max_odds = st.number_input("Maximum Odds", min_value=1.0, max_value=20.0, value=3.0)
            min_prob = st.slider("Minimum Probability", min_value=0.0, max_value=1.0, value=0.55)       

        with col3:
            st.subheader("Additional Parameters")
            # Input for date range
            start_date = st.date_input(label='Start date', 
                                       value=datetime(2024, 5, 1),
                                       min_value=datetime(2020, 1, 1),
                                       max_value=max_d)
            end_date = st.date_input(label='End date', 
                                     value=datetime(2024, 9, 1),
                                     min_value=datetime(2020, 1, 1),
                                     max_value=max_d)
            # Allow user to toggle custom stake
            custom_stake = st.checkbox("Custom Stake", value=True)

    # container to display output
    if st.button("RUN BACKTEST 🐪"):
        st.divider()
        matches['Date'] = pd.to_datetime(matches['Date'])
        filtered_matches = matches[(matches['Date'].dt.date >= start_date) & (matches['Date'].dt.date <= end_date)]
        st.markdown(f'**{len(filtered_matches)} matches** were played in the period you backtested')
        if Surface != 'All':
            filtered_matches = filtered_matches[filtered_matches['Surface'] == Surface]

        with st.spinner(text='Running Backtesting Strategy...'):
            results = backtest_strategy(filtered_matches, prob_column=prob_column, 
                                      odds_type=odds_type, criterion=criterion,
                                      stake=stake, min_ev=min_ev, min_odds=min_odds, 
                                      max_odds=max_odds, min_prob=min_prob, 
                                      Surface=Surface if Surface != 'All' else None, 
                                      custom_stake=custom_stake)

            results_df = results['matches_bet']
            results_df_subset = results_df.dropna(subset=['bet_on'])

        st.markdown(f'**{len(results_df_subset)} bets** were placed using the above strategy.')
        st.success("Backtest completed")

        # container to display numeric summaries
        with st.container(border=True):
            st.header(":green[Results]")    
            # Create a two-column layout
            cola, colb = st.columns(2)
            with cola:
                with st.container(border=True):
                    st.subheader(":red-background[Betting Summary:]")
                    metric_col, stat_col = st.columns((1, 2))
                    with metric_col:
                        st.write("Total Bets placed:")
                        st.write("Wins:")
                        st.write("Losses:")
                        st.write("Win Rate:")   
                    with stat_col:
                        st.write(f"{results['total_bets']}")
                        st.write(f"{results['wins']}")
                        st.write(f"{results['losses']}")
                        st.write(f"{results['win_rate']*100:.2f}%")
            with colb:
                with st.container(border=True):
                    st.subheader(":blue-background[Financial Summary:]")
                    metric_col, stat_col = st.columns((1, 2))
                    with metric_col:
                        st.write("Total Staked:")
                        st.write("Total Profit:")
                        st.write("ROI:")
                    with stat_col:
                        st.write(f"R{results['total_stake']:,d}.00")
                        st.write(f"R{results['total_profit']:,.2f}")
                        st.write(f"{results['roi']:.2f}%")

        # container to display visual output
        with st.expander(":red-background[EV and Implied Probability Win Rate]"):
            st.write("Left plot shows Win Rate by EV and right plot shows Baseline Probability")    
            # Create a two-column layout
            cola, colb = st.columns(2)
            with cola: 
                plot_win_rate_by_ev(results_df)
            with colb:
                plot_win_rate_by_prob(results_df)

        with st.expander(":blue-background[Profitability]"):
            st.write("Left plot shows where you made profit/loss, and that right plot shows what Surface you made money on")    
            # Create a two-column layout
            cola, colb = st.columns(2)
            with cola: 
                plot_roi_by_ev(results_df)
            with colb:
                plot_profit_by_Surface(results_df)

        with st.expander(":grey-background[Additional Graphics]"):
            st.write("Left plot shows which rounds you won/lost money on, and that right plot shows Expected value vs Actual Profit")     
            # Create a two-column layout
            cola, colb = st.columns(2)
            with cola: 
                plot_profit_by_round(results_df)
            with colb:
                ev_vs_actual_profit(results_df)  
        
        with st.expander(":grey-background[Data]"): 
            # debugging:               
            # st.dataframe(results_df)
            st.dataframe(results_df_subset)


def plot_stacked_proportion_winners_losers_binned(df, prob_column=None, use_bookmakers=False, bins=10):
    if use_bookmakers:
        # Step 1: Convert bookmaker odds to probabilities
        df['p_winner'] = 1 / df['B365W']
        df['p_loser'] = 1 / df['B365L']
        
        # Step 2: Normalize the probabilities to account for the overround
        df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + df['p_loser'])
        df['p_loser_normalized'] = 1 - df['p_winner_normalized']
        
        # Use the normalized winner probability as the primary probability for binning
        df['binned_winner_p'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
        df['binned_loser_p'] = pd.cut(df['p_loser_normalized'], bins=bins, labels=False)
    else:
        # Step 1: Bin the winner probabilities based on the provided model column
        if prob_column is None:
            raise ValueError("Please provide a valid model probability column when not using bookmakers.")
        
        df['binned_winner_p'] = pd.cut(df[prob_column], bins=bins, labels=False)
        
        # Step 2: Calculate 1 - p for losers based on model predictions
        df['loser_p'] = 1 - df[prob_column]
        df['binned_loser_p'] = pd.cut(df['loser_p'], bins=bins, labels=False)

    # Step 3: Count occurrences in each bin for winners and losers
    winners_binned = df.groupby('binned_winner_p').size()
    losers_binned = df.groupby('binned_loser_p').size()

    # Ensure that the bins align, filling in any missing bins with 0s
    bin_indices = np.arange(bins)
    winners_binned = winners_binned.reindex(bin_indices, fill_value=0)
    losers_binned = losers_binned.reindex(bin_indices, fill_value=0)

    # Step 4: Calculate proportions
    total_binned = winners_binned + losers_binned
    winners_proportion = winners_binned / total_binned
    losers_proportion = losers_binned / total_binned

    # Step 5: Plot the stacked bar chart with proportions
    plt.figure(figsize=(11, 10))

    # Plot the stacked proportions for winners and losers
    plt.bar(bin_indices, winners_proportion, color='skyblue', label='Winners (p)', width=0.8)
    plt.bar(bin_indices, losers_proportion, bottom=winners_proportion, color='salmon', label='Losers (1 - p)', width=0.8)

    # Add labels and title
    title = f'Stacked Proportion of {"Bookmaker" if use_bookmakers else "Model"} Predicted Probabilities for Winners and Losers'
    plt.title(title, fontsize=16)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    
    # Customizing the x-axis to show the bin midpoints as labels
    bin_edges = np.linspace(0, 1, bins+1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
    plt.xticks(bin_indices, [f'{round(mid, 2)}' for mid in bin_midpoints])
    
    # Add a legend
    plt.legend()
    
    # Display plot in Streamlit
    st.pyplot(plt.gcf())
    plt.close()


def plot_stacked_proportion_comparison(df, prob_column=None, bins=10):
    # Convert bookmaker odds to probabilities
    df['p_winner'] = 1 / df['B365W']
    df['p_loser'] = 1 / df['B365L']
    
    # Normalize bookmaker probabilities
    df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + df['p_loser'])
    df['p_loser_normalized'] = 1 - df['p_winner_normalized']
    
    # Bin bookmaker probabilities
    df['binned_winner_p_bookmakers'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
    df['binned_loser_p_bookmakers'] = pd.cut(df['p_loser_normalized'], bins=bins, labels=False)
    
    # Bin model probabilities
    if prob_column is None:
        raise ValueError("Please provide a valid model probability column.")
    
    df['binned_winner_p_model'] = pd.cut(df[prob_column], bins=bins, labels=False)
    df['binned_loser_p_model'] = pd.cut(1 - df[prob_column], bins=bins, labels=False)

    # Count occurrences in each bin for winners and losers
    def calculate_proportions(bin_column_winner, bin_column_loser):
        winners_binned = df.groupby(bin_column_winner).size()
        losers_binned = df.groupby(bin_column_loser).size()
        
        # Ensure that the bins align, filling in any missing bins with 0s
        bin_indices = np.arange(bins)
        winners_binned = winners_binned.reindex(bin_indices, fill_value=0)
        losers_binned = losers_binned.reindex(bin_indices, fill_value=0)
        
        # Calculate proportions
        total_binned = winners_binned + losers_binned
        winners_proportion = winners_binned / total_binned
        losers_proportion = losers_binned / total_binned
        
        return winners_proportion, losers_proportion

    # Calculate proportions for both bookmakers and models
    winners_proportion_bookmakers, losers_proportion_bookmakers = calculate_proportions('binned_winner_p_bookmakers', 'binned_loser_p_bookmakers')
    winners_proportion_model, losers_proportion_model = calculate_proportions('binned_winner_p_model', 'binned_loser_p_model')

    # Plotting stacked bar chart
    plt.figure(figsize=(12, 8))

    bin_indices = np.arange(bins)
    
    # Plot stacked proportions for bookmakers
    plt.bar(bin_indices - 0.2, winners_proportion_bookmakers, color='skyblue', label='Bookmakers Winners (p)', width=0.4)
    plt.bar(bin_indices - 0.2, losers_proportion_bookmakers, bottom=winners_proportion_bookmakers, color='salmon', label='Bookmakers Losers (1 - p)', width=0.4)

    # Plot stacked proportions for models
    plt.bar(bin_indices + 0.2, winners_proportion_model, color='lightgreen', label='Model Winners (p)', width=0.4)
    plt.bar(bin_indices + 0.2, losers_proportion_model, bottom=winners_proportion_model, color='coral', label='Model Losers (1 - p)', width=0.4)

    # Add labels and title
    plt.title('Comparison of Proportions for Bookmakers and Model Predictions', fontsize=16)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    
    # Customizing the x-axis to show the bin midpoints as labels
    bin_edges = np.linspace(0, 1, bins+1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
    plt.xticks(bin_indices, [f'{round(mid, 2)}' for mid in bin_midpoints])
    
    # Add a legend
    plt.legend()
    
    st.pyplot(plt.gcf())
    plt.close()


def plot_proportions_points(df, prob_column=None, bins=10):
    # Convert bookmaker odds to probabilities
    df['p_winner'] = 1 / df['B365W']
    df['p_loser'] = 1 / df['B365L']
    
    # Normalize bookmaker probabilities
    df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + df['p_loser'])
    df['p_loser_normalized'] = 1 - df['p_winner_normalized']
    
    # Bin bookmaker probabilities
    df['binned_winner_p_bookmakers'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
    df['binned_loser_p_bookmakers'] = pd.cut(df['p_loser_normalized'], bins=bins, labels=False)
    
    # Bin model probabilities
    if prob_column is None:
        raise ValueError("Please provide a valid model probability column.")
    
    df['binned_winner_p_model'] = pd.cut(df[prob_column], bins=bins, labels=False)
    df['binned_loser_p_model'] = pd.cut(1 - df[prob_column], bins=bins, labels=False)

    # Count occurrences in each bin for winners and losers
    def calculate_proportions(bin_column_winner, bin_column_loser):
        winners_binned = df.groupby(bin_column_winner).size()
        losers_binned = df.groupby(bin_column_loser).size()
        
        # Ensure that the bins align, filling in any missing bins with 0s
        bin_indices = np.arange(bins)
        winners_binned = winners_binned.reindex(bin_indices, fill_value=0)
        losers_binned = losers_binned.reindex(bin_indices, fill_value=0)
        
        # Calculate proportions
        total_binned = winners_binned + losers_binned
        winners_proportion = winners_binned / total_binned
        losers_proportion = losers_binned / total_binned
        
        return winners_proportion, losers_proportion

    # Calculate proportions for both bookmakers and models
    winners_proportion_bookmakers, losers_proportion_bookmakers = calculate_proportions('binned_winner_p_bookmakers', 'binned_loser_p_bookmakers')
    winners_proportion_model, losers_proportion_model = calculate_proportions('binned_winner_p_model', 'binned_loser_p_model')

    # Plotting proportions as points
    plt.figure(figsize=(14, 6))

    bin_indices = np.arange(bins)
    
    # Plot proportions for bookmakers
    plt.scatter(bin_indices - 0.2, winners_proportion_bookmakers, color='skyblue', label='Bookmakers Winners (p)', zorder=5)
    plt.scatter(bin_indices - 0.2, losers_proportion_bookmakers, color='salmon', label='Bookmakers Losers (1 - p)', zorder=5)

    # Plot proportions for models
    plt.scatter(bin_indices + 0.2, winners_proportion_model, color='lightgreen', label='Model Winners (p)', zorder=5)
    plt.scatter(bin_indices + 0.2, losers_proportion_model, color='coral', label='Model Losers (1 - p)', zorder=5)

    # Add labels and title
    plt.title('Comparison of Proportions for Bookmakers and Model Predictions (Points)', fontsize=16)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    
    # Customizing the x-axis to show the bin midpoints as labels
    bin_edges = np.linspace(0, 1, bins+1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
    plt.xticks(bin_indices, [f'{round(mid, 2)}' for mid in bin_midpoints])
    
    # Add a legend
    plt.legend()
    
    st.pyplot(plt.gcf())
    plt.close()


def plot_proportions_filled(df, prob_column=None, bins=10):
    if prob_column is None:
        raise ValueError("Please provide a valid model probability column.")
    
    # Convert bookmaker odds to probabilities
    df['p_winner'] = 1 / df['B365W']
    
    # Normalize bookmaker probabilities
    df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + (1 / df['B365L']))
    
    # Bin bookmaker and model probabilities
    df['binned_winner_p_bookmakers'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
    df['binned_winner_p_model'] = pd.cut(df[prob_column], bins=bins, labels=False)
    
    # Count occurrences in each bin for winners
    def calculate_proportions(bin_column):
        binned_counts = df.groupby(bin_column).size()
        
        # Ensure that the bins align, filling in any missing bins with 0s
        bin_indices = np.arange(bins)
        binned_counts = binned_counts.reindex(bin_indices, fill_value=0)
        
        # Calculate proportions
        proportions = binned_counts / binned_counts.sum()
        
        return proportions

    # Calculate proportions for both bookmakers and models
    proportions_bookmakers = calculate_proportions('binned_winner_p_bookmakers')
    proportions_model = calculate_proportions('binned_winner_p_model')

    # Plotting filled areas
    plt.figure(figsize=(14, 6))
    
    bin_indices = np.arange(bins)
    
    # Plot filled area for bookmakers
    plt.fill_between(bin_indices, 0, proportions_bookmakers, color='skyblue', alpha=0.5, label='Bookmakers Winners (p)')
    
    # Plot filled area for models
    plt.fill_between(bin_indices, 0, proportions_model, color='lightgreen', alpha=0.5, label='Model Winners (p)')
    
    # Add labels and title
    plt.title('Comparison of Winner Proportions for Bookmakers and Model Predictions (Filled Areas)', fontsize=16)
    plt.xlabel('Probability Bin', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    
    # Customizing the x-axis to show the bin midpoints as labels
    bin_edges = np.linspace(0, 1, bins+1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
    plt.xticks(bin_indices, [f'{round(mid, 2)}' for mid in bin_midpoints])
    
    # Add a legend
    plt.legend()
    
    st.pyplot(plt.gcf())
    plt.close()


def plot_proportion_winners_line_chart(df, prob_column=None, bins=10):
    # Check if the model probability column is provided
    if prob_column is None:
        raise ValueError("Please provide a valid model probability column.")
    
    # Convert bookmaker odds to probabilities
    df['p_winner'] = 1 / df['B365W']
    df['p_loser'] = 1 / df['B365L']
    
    # Normalize bookmaker probabilities so they sum to 1
    df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + df['p_loser'])
    df['p_loser_normalized'] = 1 - df['p_winner_normalized']
    
    # Bin bookmaker probabilities and model probabilities for winners and losers
    df['binned_winner_p_bookmakers'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
    df['binned_loser_p_bookmakers'] = pd.cut(df['p_loser_normalized'], bins=bins, labels=False)
    
    df['binned_winner_p_model'] = pd.cut(df[prob_column], bins=bins, labels=False)
    df['binned_loser_p_model'] = pd.cut(1 - df[prob_column], bins=bins, labels=False)
    
    # Count occurrences in each bin for winners and losers
    def calculate_proportions(bin_column_winner, bin_column_loser):
        winners_binned = df.groupby(bin_column_winner).size()
        losers_binned = df.groupby(bin_column_loser).size()
        
        # Ensure that the bins align, filling in any missing bins with 0s
        bin_indices = np.arange(bins)
        winners_binned = winners_binned.reindex(bin_indices, fill_value=0)
        losers_binned = losers_binned.reindex(bin_indices, fill_value=0)
        
        # Calculate proportions: winners / total (winners + losers)
        total_binned = winners_binned + losers_binned
        winners_proportion = winners_binned / total_binned
        
        return winners_proportion

    # Calculate proportions for both bookmakers and model
    winners_proportion_bookmakers = calculate_proportions('binned_winner_p_bookmakers', 'binned_loser_p_bookmakers')
    winners_proportion_model = calculate_proportions('binned_winner_p_model', 'binned_loser_p_model')
    
    # Plotting the proportions
    plt.figure(figsize=(10, 10))
    
    bin_indices = np.arange(bins)
    
    # Plot the bookmakers' proportion of winners
    plt.plot(bin_indices, winners_proportion_bookmakers, color='skyblue', label='Bookmakers Winners (p)', marker='o', linestyle='-')
    
    # Plot the model's proportion of winners
    plt.plot(bin_indices, winners_proportion_model, color='lightgreen', label=f'Model {prob_column} Winners (p)', marker='o', linestyle='-')
    
    # Add labels and title
    plt.title('Proportion of Winners in Each Probability Bin for Bookmakers and Model Predictions', fontsize=16)
    plt.xlabel('Probability Bin', fontsize=12)
    plt.ylabel('Proportion of Winners', fontsize=12)
    
    # Customize the x-axis to show the upper values of the bins
    bin_edges = np.linspace(0, 1, bins + 1)
    plt.xticks(bin_indices, [f'{round(edge, 2)}' for edge in bin_edges[1:]])
    
    # Add a legend
    plt.legend()
    plt.show()
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    plt.close()


def plot_filled_proportion_winners(df, prob_column=None, bins=10, bookmaker='B365'):
    if prob_column is None:
        raise ValueError("Please provide a valid model probability column.")
    
    # Define mappings for the bookmaker columns
    bookmaker_columns = {
        'B365': ('B365W', 'B365L'),
        'PS': ('PSW', 'PSL'),
        'best': ('MaxW', 'MaxL'),
        'average': ('AvgW', 'AvgL')
    }
    
    if bookmaker not in bookmaker_columns:
        raise ValueError(f"Invalid bookmaker option. Choose from: {list(bookmaker_columns.keys())}")
    
    winner_col, loser_col = bookmaker_columns[bookmaker]
    
    # Convert bookmaker odds to probabilities
    df['p_winner'] = 1 / df[winner_col]
    df['p_loser'] = 1 / df[loser_col]
    
    # Normalize bookmaker probabilities
    df['p_winner_normalized'] = df['p_winner'] / (df['p_winner'] + df['p_loser'])
    df['p_loser_normalized'] = 1 - df['p_winner_normalized']
    
    # Bin bookmaker probabilities and model probabilities
    df['binned_winner_p_bookmakers'] = pd.cut(df['p_winner_normalized'], bins=bins, labels=False)
    df['binned_loser_p_bookmakers'] = pd.cut(df['p_loser_normalized'], bins=bins, labels=False)
    
    df['binned_winner_p_model'] = pd.cut(df[prob_column], bins=bins, labels=False)
    df['binned_loser_p_model'] = pd.cut(1 - df[prob_column], bins=bins, labels=False)
    
    # Count occurrences in each bin for winners and losers
    def calculate_proportions(bin_column_winner, bin_column_loser):
        winners_binned = df.groupby(bin_column_winner).size()
        losers_binned = df.groupby(bin_column_loser).size()
        
        # Ensure that the bins align, filling in any missing bins with 0s
        bin_indices = np.arange(bins)
        winners_binned = winners_binned.reindex(bin_indices, fill_value=0)
        losers_binned = losers_binned.reindex(bin_indices, fill_value=0)
        
        # Calculate proportions
        total_binned = winners_binned + losers_binned
        winners_proportion = winners_binned / total_binned
        
        return winners_proportion

    # Calculate proportions for both bookmakers and model
    winners_proportion_bookmakers = calculate_proportions('binned_winner_p_bookmakers', 'binned_loser_p_bookmakers')
    winners_proportion_model = calculate_proportions('binned_winner_p_model', 'binned_loser_p_model')
    
    # Plotting the filled area
    plt.figure(figsize=(12, 11))
    
    bin_indices = np.arange(bins)
    
    # Plot filled area for bookmakers
    plt.fill_between(bin_indices, winners_proportion_bookmakers, color='skyblue', alpha=0.5, label=f'{bookmaker} Winners (p)')
    
    # Plot filled area for model
    plt.fill_between(bin_indices, winners_proportion_model, color='lightgreen', alpha=0.5, label=f'Model {prob_column} Winners (p)')
    
    # Add labels and title
    plt.title(f'Proportion of Winners in Each Probability Bin ({bookmaker} vs Model)', fontsize=16)
    plt.xlabel('Probability Bin', fontsize=12)
    plt.ylabel('Proportion of Winners', fontsize=12)
    
    # Customize the x-axis to show the upper values of the bins
    bin_edges = np.linspace(0, 1, bins + 1)
    plt.xticks(bin_indices, [f'{round(edge, 2)}' for edge in bin_edges[1:]])
    
    # Add a legend
    plt.legend()
    plt.show()
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    plt.close()


# %% tools

# Main content based on the selected tool

# player info tool
if tool == "Player Info":
    st.title(":green[Player Info Tool]")
    st.markdown("""
                The :green[Player Info Tool] offers comprehensive player analysis.   
                
                Use this tool to get basic player information in the :green-background[Biography] Tab. Match history, win percentages, and Surface performance  
                can be found in the :green-background[Statistics] Tab. View and analyze :green[Elo Rating] and  :green[ATP Ranking] history in the :green-background[Elo and Ranking] Tab.   
                
                Please note that :red-background[Betting Info] is still under development and will be added in a future release.
                
    """)
    player_info_tool(players, matches, elo_df, glicko_df)
    
# player comparison tool
elif tool == "Player Comparison":
    st.title(":green[Player Comparison Tool]")
    st.markdown("""
                The :green[Player Comparison Tool] enables side-by-side analysis of two players across five sections.
                
                Use :green-background[Biography Comparison], to view and compare basic info about each player;
                while :green-background[Statistics Comparison] lets you look more into statistics.  
                If you want to just focus on their rankings or rating performance you can use :green-background[ATP Rankings and Elo Ratings].  
                :green-background[Head-to-Head] shows you any previous match up between the two players; and finally, if you want to
                predict the outcome unmder different conditions, :green-background[Expected Outcome], provides expected probabilities.   
                
                These are all designed to visually compare performance trends and evaluate how two players stack up against each other.
                
                """)
    player_comparison_tool(players, elo_df, matches)

# match maker tool
elif tool == "Match Maker":
    st.title(":green[Match Maker Tool]")
    st.markdown("""
                The :green[Match Maker Tool] help you make informed betting decisions.
                
                This will help you make winning bets.
                
                """)
    match_maker_tool()

# odds converter tool    
elif tool == "Odds Converter":
    st.title(":green[Odds Converter]")
    st.markdown("""
                The :green[Odds Converter Tool] allows you convert different odds types. It will also give you the implied probability.
                
                You can convert between :green-background[Decimal Odds], :green-background[Fractional Odds], and :green-background[American Odds]

                On the right you can read more into :orange-background[Understanding Odds and Probability].
                
    """)
    st.divider()
    odds_tool()

# draw simulator tool    
elif tool == "Tournament Simulator":
    st.title(":green[Tournament Simulator Tool]")
    st.markdown("""
                The :green[Tournament Simulator Tool] lets you easily see how
                a tournament might play out, based on current Elo Ratings.
                
                Upload a :orange-background[.csv file] with a list of player
                names in a column called :grey-background[Player].   
                
                Examples of :green-background[VALID] names are: 
                    :green-background[Rafael Nadal],
                :green-background[Casper Ruud], :green-background[Felix
                Auger Aliassime]  
                Examples of :red-background[INVALID] names are: 
                    :red-background[Rafa Nadal],
                :red-background[C Ruud], :red-background[Felix
                Auger-Aliassime] 
    """)
    st.divider()
    tournament_draw_simulator_tool(elo_df)

# strategy backtesting tool
elif tool == 'Strategy Simulator':
    st.title(":green[Strategy Simulator and Backtesting]")
    backtest_strategy_tool(matches, max_d, backtest_strategy)
    
    

# %% about

if about is True:
    st.divider()
    st.title(":blue[About:]")
    st.write(f":grey-background[Total players in database:] {len(players)}")
    st.write(f":grey-background[Total matches in database:] {len(matches)}")
    st.write(f":grey-background[Total players with Elo calcs:] {len(elo_df)}")
    st.write(":grey-background[Lines of Code:] 3334")
    st.markdown('<a href="mailto:kaalvoetranger@gmail.com">Email bugs and suggestions to me!</a>', unsafe_allow_html=True)
    with st.container(border=True):
       
        with st.expander("Important Graphic to help understand the data", expanded = False):
            st.write("How the baseline Elo Model Predicts Winners from P")
            plot_stacked_proportion_winners_losers_binned(matches, prob_column='expected_probA', bins=20)
            st.divider()
            plot_stacked_proportion_winners_losers_binned(matches, use_bookmakers=True, bins=20)
            st.divider()
            plot_stacked_proportion_comparison(matches, prob_column='expected_probA', bins=20)
            st.divider()
            plot_proportions_points(matches, prob_column='expected_probA', bins=20)
            st.write("proportion on the y axis in the below graph is proportion of total count.")
            st.write("it's showing how the probabilities are distributed. our model gives a lot more p closer to 0.5, while the bookies can stretch out the middle")
            plot_proportions_filled(matches, prob_column='expected_probA', bins=20)
            st.divider()
            plot_proportion_winners_line_chart(matches,'expected_probA', bins=20)
            plot_filled_proportion_winners(matches, prob_column='expected_probS', bins=20, bookmaker='PS')
            plot_filled_proportion_winners(matches, prob_column='expected_probS', bins=20, bookmaker='B365')
            plot_filled_proportion_winners(matches, prob_column='expected_probS', bins=20, bookmaker='average')
            plot_filled_proportion_winners(matches, prob_column='expected_probS', bins=20, bookmaker='best')
        
        with st.expander("Important functions to understand", expanded=False):
            st.write("EV calc")
            st.code("""
                    def calculate_ev(expected_prob, odds, stake):
                        # Calculate the implied probability from the bookmaker's odds
                        implied_prob = 1 / odds
                        
                        # Calculate potential profit (odds - 1) * stake
                        profit = (odds - 1) * stake
                        
                        # Calculate EV
                        ev = (expected_prob * profit) - ((1 - expected_prob) * stake)
                        
                        return ev                
                    """)
            st.write("Probabilities from Elo")
            st.code("""
                    def expected_outcome(player1, player2, Surface, weight_Surface=0.9, h2h_weight=10):
                        ""
                        Calculate the expected outcome probabilities for a match between two players.

                        Parameters:
                        player1 (str): Name of the first player.
                        player2 (str): Name of the second player.
                        Surface (str): The Surface type ('Grass', 'Clay', 'Hard').
                        weight_Surface (float): Weight given to the Surface-specific Elo rating (default is 0.9).
                        h2h_weight (int): Weight applied to the head-to-head record (default is 10).
                        
                        Returns:
                        tuple: A tuple containing the expected probabilities for Player 1 and Player 2 based on 
                               overall Elo (probA for Player 1, probA_2 for Player 2), Surface Elo (probS for Player 1, probS_2 for Player 2), 
                               and combined Elo including head-to-head history (probH for Player 1, probH_2 for Player 2).
                        ""
                        weight_all = 1 - weight_Surface
                        
                        # Retrieve Elo ratings for both players
                        elo1_all = elo_ratings_all[player1]
                        elo2_all = elo_ratings_all[player2]
                        
                        # Determine Surface-specific Elo ratings
                        if Surface == "Clay":
                            elo1_Surface = elo_ratings_clay[player1]
                            elo2_Surface = elo_ratings_clay[player2]
                        elif Surface == "Hard":
                            elo1_Surface = elo_ratings_hard[player1]
                            elo2_Surface = elo_ratings_hard[player2]
                        elif Surface == "Grass":
                            elo1_Surface = elo_ratings_grass[player1]
                            elo2_Surface = elo_ratings_grass[player2]
                        else:
                            elo1_Surface = elo_ratings_all[player1]
                            elo2_Surface = elo_ratings_all[player2]
                        
                        # Calculate combined Elo for both players (weighted average of overall and Surface-specific ratings)
                        combined_elo1 = weight_all * elo1_all + weight_Surface * elo1_Surface
                        combined_elo2 = weight_all * elo2_all + weight_Surface * elo2_Surface
                        
                        # Calculate expected probabilities based on overall Elo
                        expected_probA = 1 / (1 + 10 ** ((elo2_all - elo1_all) / 400))
                        expected_probA_2 = 1 - expected_probA
                        
                        # Calculate expected probabilities based on Surface Elo (combined Elo)
                        expected_probS = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
                        expected_probS_2 = 1 - expected_probS
                        
                        # Adjust for head-to-head records
                        if (player1, player2) in head_to_head:
                            h2h_record = head_to_head[(player1, player2)]
                            player1_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
                            combined_elo1 += h2h_weight * player1_h2h_advantage
                        
                        if (player2, player1) in head_to_head:
                            h2h_record = head_to_head[(player2, player1)]
                            player2_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
                            combined_elo2 += h2h_weight * player2_h2h_advantage
                        
                        # Calculate expected probabilities considering head-to-head adjustments
                        expected_probH = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
                        expected_probH_2 = 1 - expected_probH
                        
                        # Return expected probabilities for both Player 1 and Player 2
                        return expected_probA, expected_probS, expected_probH, expected_probA_2, expected_probS_2, expected_probH_2

                    """)
        
        with st.expander("What the data looks like in the backend:", expanded=False):
            
            st.subheader("Few rows of 'matches' dataframe")
            st.dataframe(matches.tail())
            st.subheader("Few rows of 'players' dataframe")
            st.dataframe(players.head())
            st.divider()
            st.subheader("Few rows of 'Elo' dataframe")
            st.dataframe(elo_df.head(50))
        
       
