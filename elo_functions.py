#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% 1 dependencies

import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
from datetime import datetime 
import streamlit as st

import warnings
# Ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set directories and ignore warnings (edit this sections when deploying)
base_dir = os.path.expanduser("~/Desktop/tennis_app/")
data_dir = os.path.join(base_dir, "datasets/")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

matches = pd.read_csv(data_dir + "matches.csv")
players = pd.read_csv(data_dir + "atp_players.csv")
matches['tourney_date'] = pd.to_datetime(matches['tourney_date'])

# %% 2 elo functions


def initialize_elo_ratings():
    """
    Initialize Elo ratings for all players across different surfaces and other related statistics.
    
    Creates global variables for Elo ratings on grass, clay, and hard courts, as well as for 
    overall Elo ratings. Each player starts with an Elo rating of 1500, and match history and head-to-head 
    statistics are also initialized.
    """
    global K, starting_elo, elo_ratings_all, elo_ratings_grass, elo_ratings_clay, elo_ratings_hard, most_recent_date, match_count, matches_won, head_to_head
    K = 32
    players = set(matches['winner_name']).union(set(matches['loser_name']))
    starting_elo = 1500  # Default starting Elo rating
    elo_ratings_all = {}
    elo_ratings_grass = {}
    elo_ratings_clay = {}
    elo_ratings_hard = {}
    
    most_recent_date = {}
    match_count = {}
    matches_won = {}
    head_to_head = {}
    
    for player in players:
        if player not in elo_ratings_all:
            elo_ratings_all[player] = starting_elo
        if player not in elo_ratings_clay:
            elo_ratings_clay[player] = starting_elo
        if player not in elo_ratings_hard:
            elo_ratings_hard[player] = starting_elo
        if player not in elo_ratings_grass:
            elo_ratings_grass[player] = starting_elo   


def get_elo(player, elo_dict):
    """
    Retrieve the Elo rating of a player from the given Elo dictionary. 
    If the player does not exist, return 'Value'.
    
    Parameters:
    player (str): Name of the player.
    elo_dict (dict): Dictionary containing Elo ratings for players.
    
    Returns:
    float or str: The player's Elo rating or 'Value' if not found.
    """
    return elo_dict.setdefault(player, 'Value')


def update_elo(winner, loser, elo_dict, K=32):
    """
    Update the Elo ratings for the winner and loser after a match.
    
    Parameters:
    winner (str): Name of the winning player.
    loser (str): Name of the losing player.
    elo_dict (dict): Dictionary containing the Elo ratings of players.
    K (int): The K-factor, determining the sensitivity of Elo updates (default is 32).
    
    Updates the Elo ratings of both players based on the outcome.
    """
    winner_elo = get_elo(winner, elo_dict)
    loser_elo = get_elo(loser, elo_dict)
    
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    
    elo_dict[winner] = winner_elo + K * (1 - expected_winner)
    elo_dict[loser] = loser_elo + K * (0 - expected_loser)
    

def expected_outcome(player1, player2, surface, weight_surface=0.9, h2h_weight=10):
    """
    Calculate the expected outcome probabilities for a match between two players.

    Parameters:
    player1 (str): Name of the first player.
    player2 (str): Name of the second player.
    surface (str): The surface type ('Grass', 'Clay', 'Hard').
    weight_surface (float): Weight given to the surface-specific Elo rating (default is 0.9).
    h2h_weight (int): Weight applied to the head-to-head record (default is 10).
    
    Returns:
    tuple: A tuple containing the expected probabilities for Player 1 and Player 2 based on 
           overall Elo (probA for Player 1, probA_2 for Player 2), surface Elo (probS for Player 1, probS_2 for Player 2), 
           and combined Elo including head-to-head history (probH for Player 1, probH_2 for Player 2).
    """
    weight_all = 1 - weight_surface
    
    # Retrieve Elo ratings for both players
    elo1_all = elo_ratings_all[player1]
    elo2_all = elo_ratings_all[player2]
    
    # Determine surface-specific Elo ratings
    if surface == "Clay":
        elo1_surface = elo_ratings_clay[player1]
        elo2_surface = elo_ratings_clay[player2]
    elif surface == "Hard":
        elo1_surface = elo_ratings_hard[player1]
        elo2_surface = elo_ratings_hard[player2]
    elif surface == "Grass":
        elo1_surface = elo_ratings_grass[player1]
        elo2_surface = elo_ratings_grass[player2]
    else:
        elo1_surface = elo_ratings_all[player1]
        elo2_surface = elo_ratings_all[player2]
    
    # Calculate combined Elo for both players (weighted average of overall and surface-specific ratings)
    combined_elo1 = weight_all * elo1_all + weight_surface * elo1_surface
    combined_elo2 = weight_all * elo2_all + weight_surface * elo2_surface
    
    # Calculate expected probabilities based on overall Elo
    expected_probA = 1 / (1 + 10 ** ((elo2_all - elo1_all) / 400))
    expected_probA_2 = 1 - expected_probA
    
    # Calculate expected probabilities based on surface Elo (combined Elo)
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


def update_head_to_head(winner, loser):
    """
    Update the head-to-head record between two players after a match.
    
    Parameters:
    winner (str): Name of the winning player.
    loser (str): Name of the losing player.
    
    Updates the wins and losses between the players in the head-to-head dictionary.
    """
    head_to_head.setdefault((winner, loser), {'wins': 0, 'losses': 0})
    head_to_head.setdefault((loser, winner), {'wins': 0, 'losses': 0})
    
    head_to_head[(winner, loser)]['wins'] += 1
    head_to_head[(loser, winner)]['losses'] += 1
    
    
def get_rating(player):
    """
    Retrieve the overall Elo rating for a given player.
    
    Parameters:
    player (str): Name of the player.
    
    Returns:
    float or str: The player's Elo rating or 'Value' if not found.
    """
    return elo_ratings_all.get(player, 'Value')


def simulate_match(player1, player2, player1_wins, surface):
    """
    Simulate a match between two players, update Elo ratings, and adjust head-to-head records.
    
    Parameters:
    player1 (str): Name of the first player.
    player2 (str): Name of the second player.
    player1_wins (bool): True if player 1 wins, False if player 2 wins.
    surface (str): The surface on which the match was played ('Grass', 'Clay', 'Hard').
    
    Updates Elo ratings and head-to-head records based on the match outcome.
    """
    if player1_wins:
        update_elo(player1, player2, elo_ratings_all)
        if surface == 'Grass':
            update_elo(player1, player2, elo_ratings_grass)
        elif surface == 'Clay':
            update_elo(player1, player2, elo_ratings_clay)
        elif surface == 'Hard':
            update_elo(player1, player2, elo_ratings_hard)
        update_head_to_head(player1, player2)
    else:
        update_elo(player2, player1, elo_ratings_all)
        if surface == 'Grass':
            update_elo(player2, player1, elo_ratings_grass)
        elif surface == 'Clay':
            update_elo(player2, player1, elo_ratings_clay)
        elif surface == 'Hard':
            update_elo(player2, player1, elo_ratings_hard)
        update_head_to_head(player2, player1)


def print_elo_ratings():
    """
    Print the Elo ratings of all players for each surface (overall, grass, clay, hard).
    
    Displays the Elo ratings for all players in the Elo dictionaries.
    """
    print("Overall Elo Ratings:")
    for player, rating in elo_ratings_all.items():
        print(f"{player}: {rating}")
    print("Grass Elo Ratings:")
    for player, rating in elo_ratings_grass.items():
        print(f"{player}: {rating}")
    print("Clay Elo Ratings:")
    for player, rating in elo_ratings_clay.items():
        print(f"{player}: {rating}")
    print("Hard Elo Ratings:")
    for player, rating in elo_ratings_hard.items():
        print(f"{player}: {rating}")


def calculate_and_analyze_elo():
    """
    Initialize Elo ratings, simulate matches, update Elo ratings, and analyze results.
    
    Simulates matches, calculates expected probabilities, updates player Elo ratings, and analyzes the 
    results by calculating accuracy and log loss.
    
    Returns:
    None
    """
    global elo_df
    initialize_elo_ratings()
    
    # Initialize columns for Elo and expected probability
    matches['elo_winner_before'] = 0
    matches['elo_winner_after'] = 0
    matches['expected_probA'] = 0.0
    matches['expected_probS'] = 0.0
    matches['expected_probH'] = 0.0
    matches['expected_probA_2'] = 0.0
    matches['expected_probS_2'] = 0.0
    matches['expected_probH_2'] = 0.0
    matches['elo_loser_before'] = 0
    matches['elo_loser_after'] = 0
    
    # Initialize lists for storing actual results and predicted probabilities
    actual_results = []
    predicted_probs = []

    # Initialize dictionaries to store the most recent match date and ranking
    most_recent_date = {}
    most_recent_ranking = {}

    for index, row in matches.iterrows():
        player1 = row['winner_name']
        player2 = row['loser_name']
        surface = row['surface']
        match_date = row['tourney_date']
        ranking_winner = row['WRank']
        ranking_loser = row['LRank']
        
        # Ensure both players have an Elo rating
        if player1 not in elo_ratings_all:
            elo_ratings_all[player1] = 1500
        if player2 not in elo_ratings_all:
            elo_ratings_all[player2] = 1500
        
        # Store current Elo ratings before the match
        elo_winner_before = round(elo_ratings_all[player1], 1)
        elo_loser_before = round(elo_ratings_all[player2], 1)
        matches.loc[index, 'elo_winner_before'] = elo_winner_before
        matches.loc[index, 'elo_loser_before'] = elo_loser_before
        
        # Calculate the expected outcome
        expected_probA = expected_outcome(player1, player2, surface)[0]
        expected_probS = expected_outcome(player1, player2, surface)[1]
        expected_probH = expected_outcome(player1, player2, surface)[2]
        matches.loc[index, 'expected_probA'] = expected_probA
        matches.loc[index, 'expected_probS'] = expected_probS
        matches.loc[index, 'expected_probH'] = expected_probH
        matches.loc[index, 'expected_probA_2'] = 1 - expected_probA
        matches.loc[index, 'expected_probS_2'] = 1 - expected_probS
        matches.loc[index, 'expected_probH_2'] = 1 - expected_probH
        
        # Determine who won and simulate the match
        player1_wins = row['winner_name'] == player1
        simulate_match(player1, player2, player1_wins, surface)
        
        # Update the Elo ratings after the match
        elo_winner_after = round(elo_ratings_all[player1], 1)
        elo_loser_after = round(elo_ratings_all[player2], 1)
        matches.loc[index, 'elo_winner_after'] = elo_winner_after
        matches.loc[index, 'elo_loser_after'] = elo_loser_after

        # Update the most recent match date and ranking for both players
        if player1 in most_recent_date:
            if match_date > most_recent_date[player1]:
                most_recent_date[player1] = match_date
                most_recent_ranking[player1] = ranking_winner
        else:
            most_recent_date[player1] = match_date
            most_recent_ranking[player1] = ranking_winner

        if player2 in most_recent_date:
            if match_date > most_recent_date[player2]:
                most_recent_date[player2] = match_date
                most_recent_ranking[player2] = ranking_loser
        else:
            most_recent_date[player2] = match_date
            most_recent_ranking[player2] = ranking_loser
        
        # Update match count and matches won for both players
        match_count[player1] = match_count.get(player1, 0) + 1
        match_count[player2] = match_count.get(player2, 0) + 1
        matches_won[player1] = matches_won.get(player1, 0) + 1

        # Append actual result and predicted probability
        actual_results.append(1 if player1_wins else 0)
        predicted_probs.append(expected_probA)

    # Create DataFrames from the dictionaries
    recent_date_df = pd.DataFrame.from_dict(most_recent_date, orient='index', columns=['Most_Recent_Date'])
    recent_ranking_df = pd.DataFrame.from_dict(most_recent_ranking, orient='index', columns=['Most_Recent_Ranking'])
    match_count_df = pd.DataFrame.from_dict(match_count, orient='index', columns=['Match_Count'])
    matches_won_df = pd.DataFrame.from_dict(matches_won, orient='index', columns=['Matches_Won'])        
 
    # Calculate win percentage and add it to the matches_won_df DataFrame
    matches_won_df['Win_Percentage'] = matches_won_df.index.map(
        lambda player: matches_won.get(player, 0) / match_count.get(player, 1) * 100)
        
    # Combine all Elo ratings into a single DataFrame
    elo_all_df = pd.DataFrame.from_dict(elo_ratings_all, orient='index', columns=['Elo_ALL'])
    elo_grass_df = pd.DataFrame.from_dict(elo_ratings_grass, orient='index', columns=['Elo_Grass'])
    elo_clay_df = pd.DataFrame.from_dict(elo_ratings_clay, orient='index', columns=['Elo_Clay'])
    elo_hard_df = pd.DataFrame.from_dict(elo_ratings_hard, orient='index', columns=['Elo_Hard'])

    # Merge the dataframes to create a consolidated view of all Elo ratings, most recent date, match count, and win percentage
    elo_combined_df = elo_all_df.join([elo_grass_df, elo_clay_df, elo_hard_df, recent_date_df, recent_ranking_df, match_count_df, matches_won_df])

    # Sort the dataframe by overall Elo rating (optional)
    elo_combined_df.sort_values(by='Elo_ALL', ascending=False, inplace=True)
    elo_df = elo_combined_df.copy()
    elo_df['Elo_ALL'] = elo_df['Elo_ALL'].round(0).astype(int)
    elo_df['Elo_Hard'] = elo_df['Elo_Hard'].round(0).astype(int)
    elo_df['Elo_Clay'] = elo_df['Elo_Clay'].round(0).astype(int)
    elo_df['Elo_Grass'] = elo_df['Elo_Grass'].round(0).astype(int)
    
    # Fill NA values if any
    elo_df[['Matches_Won', 'Win_Percentage']] = elo_df[['Matches_Won', 'Win_Percentage']].fillna(0) 
    
    # Calculate accuracy and log loss
    correct_predictions = sum(1 for prob in predicted_probs if prob >= 0.5)
    accuracy = correct_predictions / len(predicted_probs) * 100
    
    # Manually calculate log loss
    predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)  # Avoid log(0) errors
    log_loss_value = -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for y, p in zip(actual_results, predicted_probs)])
    
    print("Elo Calculations completed...")
    print(f"Accuracy of Baseline Model: {accuracy:.2f}%")
    print(f"Log Loss of Baseline Model: {log_loss_value:.4f}")
    return elo_df, matches


# %% 3 testing / debugging

#elo_df, matches = calculate_and_analyze_elo()
