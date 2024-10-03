#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:23:49 2024

@author: kaalvoetranger
"""
# %% import


import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
from datetime import datetime 
import streamlit as st
import math
from datetime import timedelta

import warnings
# Ignore specific types of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#%% local variables (these are generated in main.py)
app_dir = os.getcwd()
data_dir = os.path.expanduser("~app_dir/datasets/")
g_url1 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/matches_v1.csv'
g_url2 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/atp_players_v1.csv'
matches = pd.read_csv(g_url1)
players = pd.read_csv(g_url2)
    
matches['Date'] = pd.to_datetime(matches['Date'])
players['dob'] = pd.to_datetime(players['dob'])

# %% class EloFunctions


class EloFunctions:
    def __init__(self, matches=None, players=None, K=32):
        """
        Initialize EloFunctions with necessary datasets and default settings.
        
        Args:
            matches (DataFrame): DataFrame containing match results.
            K (int): K-factor for Elo rating calculations.
        """
        if players is None or matches is None:
            raise ValueError("Error: 'players' and 'matches' DataFrames not in environment")

        self.matches = matches
        self.players = players
        self.K = K
        self.starting_elo = 1500  # Default starting Elo rating
        self.elo_ratings_all = {}
        self.elo_ratings_grass = {}
        self.elo_ratings_clay = {}
        self.elo_ratings_hard = {}
        self.most_recent_date = {}
        self.match_count = {}
        self.matches_won = {}
        self.head_to_head = {}
        self.elo_df = pd.DataFrame()

    def initialize_elo_ratings(self):
        """
        Initialize Elo ratings for all players across different surfaces 
        and other related statistics.
        """
        players = set(self.matches['Winner']).union(set(self.matches['Loser']))

        for player in players:
            self.elo_ratings_all[player] = self.starting_elo
            self.elo_ratings_clay[player] = self.starting_elo
            self.elo_ratings_hard[player] = self.starting_elo
            self.elo_ratings_grass[player] = self.starting_elo   

    def get_elo(self, player, elo_dict):
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

    def update_elo(self, winner, loser, elo_dict):
        """
        Update the Elo ratings for the winner and loser after a match.
        
        Parameters:
        winner (str): Name of the winning player.
        loser (str): Name of the losing player.
        elo_dict (dict): Dictionary containing the Elo ratings of players.
        
        Updates the Elo ratings of both players based on the outcome.
        """
        winner_elo = self.get_elo(winner, elo_dict)
        loser_elo = self.get_elo(loser, elo_dict)
        
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
        
        elo_dict[winner] = winner_elo + self.K * (1 - expected_winner)
        elo_dict[loser] = loser_elo + self.K * (0 - expected_loser)    
        
    def expected_outcome(self, player1, player2, Surface, weight_Surface=0.9, h2h_weight=10):
        """
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
        """
        weight_all = 1 - weight_Surface
        
        # Retrieve Elo ratings for both players
        elo1_all = self.elo_ratings_all[player1]
        elo2_all = self.elo_ratings_all[player2]
        
        # Determine Surface-specific Elo ratings
        if Surface == "Clay":
            elo1_Surface = self.elo_ratings_clay[player1]
            elo2_Surface = self.elo_ratings_clay[player2]
        elif Surface == "Hard":
            elo1_Surface = self.elo_ratings_hard[player1]
            elo2_Surface = self.elo_ratings_hard[player2]
        elif Surface == "Grass":
            elo1_Surface = self.elo_ratings_grass[player1]
            elo2_Surface = self.elo_ratings_grass[player2]
        else:
            elo1_Surface = self.elo_ratings_all[player1]
            elo2_Surface = self.elo_ratings_all[player2]
        
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
        if (player1, player2) in self.head_to_head:
            h2h_record = self.head_to_head[(player1, player2)]
            player1_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
            combined_elo1 += h2h_weight * player1_h2h_advantage
        
        if (player2, player1) in self.head_to_head:
            h2h_record = self.head_to_head[(player2, player1)]
            player2_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
            combined_elo2 += h2h_weight * player2_h2h_advantage
        
        # Calculate expected probabilities considering head-to-head adjustments
        expected_probH = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
        expected_probH_2 = 1 - expected_probH
        
        # Return expected probabilities for both Player 1 and Player 2
        return expected_probA, expected_probS, expected_probH, expected_probA_2, expected_probS_2, expected_probH_2
    
    def update_head_to_head(self, winner, loser):
        """
        Update the head-to-head record between two players after a match.
        
        Parameters:
        winner (str): Name of the winning player.
        loser (str): Name of the losing player.
        
        Updates the wins and losses between the players in the head-to-head dictionary.
        """
        self.head_to_head.setdefault((winner, loser), {'wins': 0, 'losses': 0})
        self.head_to_head.setdefault((loser, winner), {'wins': 0, 'losses': 0})
        
        self.head_to_head[(winner, loser)]['wins'] += 1
        self.head_to_head[(loser, winner)]['losses'] += 1
        
    def get_rating(self, player):
        """
        Retrieve the overall Elo rating for a given player.
        
        Parameters:
        player (str): Name of the player.
        
        Returns:
        float or str: The player's Elo rating or 'Value' if not found.
        """
        return self.elo_ratings_all.get(player, 'Value')

    def simulate_match(self, player1, player2, player1_wins, Surface):
        """
        Simulate a match between two players, update Elo ratings, and adjust head-to-head records.
        
        Parameters:
        player1 (str): Name of the first player.
        player2 (str): Name of the second player.
        player1_wins (bool): True if player 1 wins, False if player 2 wins.
        Surface (str): The Surface on which the match was played ('Grass', 'Clay', 'Hard').
        
        Updates Elo ratings and head-to-head records based on the match outcome.
        """
        if player1_wins:
            self.update_elo(player1, player2, self.elo_ratings_all)
            if Surface == 'Grass':
                self.update_elo(player1, player2, self.elo_ratings_grass)
            elif Surface == 'Clay':
                self.update_elo(player1, player2, self.elo_ratings_clay)
            elif Surface == 'Hard':
                self.update_elo(player1, player2, self.elo_ratings_hard)
            self.update_head_to_head(player1, player2)
        else:
            self.update_elo(player2, player1, self.elo_ratings_all)
            if Surface == 'Grass':
                self.update_elo(player2, player1, self.elo_ratings_grass)
            elif Surface == 'Clay':
                self.update_elo(player2, player1, self.elo_ratings_clay)
            elif Surface == 'Hard':
                self.update_elo(player2, player1, self.elo_ratings_hard)
            self.update_head_to_head(player2, player1)

    def print_elo_ratings(self):
        """
        Print the Elo ratings of all players for each Surface (overall, grass, clay, hard).
        
        Displays the Elo ratings for all players in the Elo dictionaries.
        """
        print("Overall Elo Ratings:")
        for player, rating in self.elo_ratings_all.items():
            print(f"{player}: {rating}")
        print("Grass Elo Ratings:")
        for player, rating in self.elo_ratings_grass.items():
            print(f"{player}: {rating}")
        print("Clay Elo Ratings:")
        for player, rating in self.elo_ratings_clay.items():
            print(f"{player}: {rating}")
        print("Hard Elo Ratings:")
        for player, rating in self.elo_ratings_hard.items():
            print(f"{player}: {rating}")
    
    def calculate_and_analyze_elo(self):
        """
        Initialize Elo ratings, simulate matches, update Elo ratings, and analyze results.
        
        Simulates matches, calculates expected probabilities, updates player Elo ratings, and analyzes the 
        results by calculating accuracy and log loss.
        
        Returns:
        DataFrame: A DataFrame with updated Elo ratings and match data (matches_elo).
        DataFrame: A DataFrame with consolidated player Elo ratings (elo_combined_df).
        """
        self.initialize_elo_ratings()
        
        # Create a copy of matches to avoid modifying the original DataFrame
        self.matches_elo = self.matches.copy()
        
        # Initialize columns for Elo and expected probability
        self.matches_elo['elo_winner_before'] = 0
        self.matches_elo['elo_winner_after'] = 0
        self.matches_elo['expected_probA'] = 0.0
        self.matches_elo['expected_probS'] = 0.0
        self.matches_elo['expected_probH'] = 0.0
        self.matches_elo['expected_probA_2'] = 0.0
        self.matches_elo['expected_probS_2'] = 0.0
        self.matches_elo['expected_probH_2'] = 0.0
        self.matches_elo['elo_loser_before'] = 0
        self.matches_elo['elo_loser_after'] = 0
        
        actual_results = []
        predicted_probs = []
        most_recent_date = {}
        most_recent_ranking = {}
    
        for index, row in self.matches_elo.iterrows():
            player1 = row['Winner']
            player2 = row['Loser']
            Surface = row['Surface']
            match_date = row['Date']
            ranking_winner = row['WRank']
            ranking_loser = row['LRank']
            
            # Ensure both players have an Elo rating
            if player1 not in self.elo_ratings_all:
                self.elo_ratings_all[player1] = 1500
            if player2 not in self.elo_ratings_all:
                self.elo_ratings_all[player2] = 1500
            
            # Store current Elo ratings before the match
            elo_winner_before = round(self.elo_ratings_all[player1], 1)
            elo_loser_before = round(self.elo_ratings_all[player2], 1)
            self.matches_elo.loc[index, 'elo_winner_before'] = elo_winner_before
            self.matches_elo.loc[index, 'elo_loser_before'] = elo_loser_before
            
            # Calculate the expected outcome
            expected_probA = self.expected_outcome(player1, player2, Surface)[0]
            expected_probS = self.expected_outcome(player1, player2, Surface)[1]
            expected_probH = self.expected_outcome(player1, player2, Surface)[2]
            self.matches_elo.loc[index, 'expected_probA'] = expected_probA
            self.matches_elo.loc[index, 'expected_probS'] = expected_probS
            self.matches_elo.loc[index, 'expected_probH'] = expected_probH
            self.matches_elo.loc[index, 'expected_probA_2'] = 1 - expected_probA
            self.matches_elo.loc[index, 'expected_probS_2'] = 1 - expected_probS
            self.matches_elo.loc[index, 'expected_probH_2'] = 1 - expected_probH
            
            # Determine who won and simulate the match
            player1_wins = row['Winner'] == player1
            self.simulate_match(player1, player2, player1_wins, Surface)
            
            # Update the Elo ratings after the match
            elo_winner_after = round(self.elo_ratings_all[player1], 1)
            elo_loser_after = round(self.elo_ratings_all[player2], 1)
            self.matches_elo.loc[index, 'elo_winner_after'] = elo_winner_after
            self.matches_elo.loc[index, 'elo_loser_after'] = elo_loser_after
    
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
            self.match_count[player1] = self.match_count.get(player1, 0) + 1
            self.match_count[player2] = self.match_count.get(player2, 0) + 1
            self.matches_won[player1] = self.matches_won.get(player1, 0) + 1
    
            # Append actual result and predicted probability
            actual_results.append(1 if player1_wins else 0)
            predicted_probs.append(expected_probA)
    
        # Create DataFrames from the dictionaries
        recent_date_df = pd.DataFrame.from_dict(most_recent_date, orient='index', columns=['Most_Recent_Date'])
        recent_ranking_df = pd.DataFrame.from_dict(most_recent_ranking, orient='index', columns=['Most_Recent_Ranking'])
        match_count_df = pd.DataFrame.from_dict(self.match_count, orient='index', columns=['Match_Count'])
        matches_won_df = pd.DataFrame.from_dict(self.matches_won, orient='index', columns=['Matches_Won'])        
     
        # Calculate win percentage and add it to the matches_won_df DataFrame
        matches_won_df['Win_Percentage'] = matches_won_df.index.map(
            lambda player: self.matches_won.get(player, 0) / self.match_count.get(player, 1) * 100)
            
        # Combine all Elo ratings into a single DataFrame
        elo_all_df = pd.DataFrame.from_dict(self.elo_ratings_all, orient='index', columns=['Elo_ALL'])
        elo_grass_df = pd.DataFrame.from_dict(self.elo_ratings_grass, orient='index', columns=['Elo_Grass'])
        elo_clay_df = pd.DataFrame.from_dict(self.elo_ratings_clay, orient='index', columns=['Elo_Clay'])
        elo_hard_df = pd.DataFrame.from_dict(self.elo_ratings_hard, orient='index', columns=['Elo_Hard'])
    
        # Merge the dataframes to create a consolidated view of all Elo ratings, most recent date, match count, and win percentage
        elo_combined_df = elo_all_df.join([elo_grass_df, elo_clay_df, elo_hard_df, recent_date_df, recent_ranking_df, match_count_df, matches_won_df])
    
        # Sort the dataframe by overall Elo rating (optional)
        elo_combined_df.sort_values(by='Elo_ALL', ascending=False, inplace=True)
        self.elo_df = elo_combined_df.copy()
        self.elo_df['Elo_ALL'] = self.elo_df['Elo_ALL'].round(0).astype(int)
        self.elo_df['Elo_Hard'] = self.elo_df['Elo_Hard'].round(0).astype(int)
        self.elo_df['Elo_Clay'] = self.elo_df['Elo_Clay'].round(0).astype(int)
        self.elo_df['Elo_Grass'] = self.elo_df['Elo_Grass'].round(0).astype(int)
        
        # Fill NA values if any
        self.elo_df[['Matches_Won', 'Win_Percentage']] = self.elo_df[['Matches_Won', 'Win_Percentage']].fillna(0) 
        
        # Calculate accuracy and log loss
        correct_predictions = sum(1 for prob in predicted_probs if prob >= 0.5)
        accuracy = correct_predictions / len(predicted_probs) * 100
        
        # Manually calculate log loss
        predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)  # Avoid log(0) errors
        log_loss_value = -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for y, p in zip(actual_results, predicted_probs)])
        
        print("Elo Calculations completed...")
        print(f"Accuracy of Baseline Model: {accuracy:.2f}%")
        print(f"Log Loss of Baseline Model: {log_loss_value:.4f}")
        
        return self.elo_df, self.matches_elo

    def get_elo_data(self):
            """
            Returns the Elo ratings and matches DataFrame.
    
            Returns:
            tuple: (elo_df, matches DataFrame)
            """
            if self.elo_df.empty:
                raise ValueError("Error: Elo ratings have not been calculated. Please run 'calculate_and_analyze_elo' first.")
            return self.elo_df, self.matches    
        
# %% debugging elo class
"""
# Initialize the class with players and matches DataFrames
elo_functions = EloFunctions(players=players, matches=matches, K=32)

# Run the Elo calculation
elo_df, matches_elo = elo_functions.calculate_and_analyze_elo()

# Later, if you want to access the Elo and matches DataFrames again:
#elo_df, matches_elo = elo_functions.get_elo_data()        
"""
# %% class GlickoFunctions


class GlickoFunctions:
    def __init__(self, matches=None, players=None, K=16, c=60, decay=0.875):
        """
        Initialize Glicko ratings and related statistics for all players.

        Args:
        matches (DataFrame): DataFrame containing match results with 'Winner' and 'Loser' columns.
        K (int): The K-factor for Glicko rating adjustments (default is 16).
        """
        if players is None or matches is None:
            raise ValueError("Error: 'players' and 'matches' DataFrames not in environment")
            
        self.K = K
        self.c = c
        self.decay = decay
        self.matches = matches
        self.starting_rating = 1500
        self.starting_rd = 325
        self.starting_volatility = 0.06
        
        # Initialize dictionaries to store Glicko ratings, RD, volatility, and match stats
        self.glicko_ratings = {}
        self.glicko_rd = {}
        self.glicko_volatility = {}
        self.last_played = {}
        self.match_count = {}
        self.matches_won = {}
        self.head_to_head = {}

        # Initialize players in the matches DataFrame
        self.initialize_players()

    def initialize_players(self):
        """Initialize Glicko ratings for all players based on the matches DataFrame."""
        players = set(self.matches['Winner']).union(set(self.matches['Loser']))
        for player in players:
            self.glicko_ratings[player] = self.starting_rating
            self.glicko_rd[player] = self.starting_rd
            self.glicko_volatility[player] = self.starting_volatility
            self.last_played[player] = None  # Initialize last played date as None

    def update_rd_for_inactivity(self, player, last_played_date, current_date, c, DELTA):
        """
        Update the rating deviation for inactivity based on the time since the last match.
    
        Parameters:
        player (str): The name of the player.
        last_played_date (pd.Timestamp): The date of the last match played by the player.
        current_date (pd.Timestamp): The current date for comparison.
        c (float): The constant to control RD increase.
        DELTA (timedelta): The time delta for determining inactivity.
        """
        # Calculate the difference in time
        time_diff = current_date - last_played_date
    
        # If the time difference exceeds DELTA, increase the RD
        if time_diff > DELTA:
            self.glicko_rd[player] = min(350, self.glicko_rd[player] + c * (time_diff.total_seconds() / (86400)))

    def update_volatility(self, player):
        """
        Update the volatility for a player based on their performance.

        If a player shows inconsistency in their performance, their volatility 
        is increased to reflect that uncertainty.

        Parameters:
        - player (str): The name of the player whose volatility is being updated.
        """
        self.glicko_volatility[player] = min(0.1, self.glicko_volatility[player] * 1.05)            

    def update_glicko(self, winner, loser, match_date, K=16, decay=0.8):
        """
        Update Glicko ratings for the winner and loser after a match.

        Parameters:
        winner (str): The name of the winning player.
        loser (str): The name of the losing player.
        match_date (datetime): The date of the match.
        K (float): The weight of the match outcome. Default is 16.
        decay (float): The decay factor for the rating deviation. Default is 0.8.
        """
        # Get current ratings and RD
        r1, rd1, v1 = self.glicko_ratings[winner], self.glicko_rd[winner], self.glicko_volatility[winner]
        r2, rd2, v2 = self.glicko_ratings[loser], self.glicko_rd[loser], self.glicko_volatility[loser]
        
        Q = math.log(10) / 400  # Reduce the denominator to increase the impact of rating changes
        
        # Calculate g(RD) for each player
        g1 = 1 / math.sqrt(1 + (3 * Q**2 * rd1**2) / (math.pi**2))
        g2 = 1 / math.sqrt(1 + (3 * Q**2 * rd2**2) / (math.pi**2))
        
        # Expected outcomes E1 and E2
        E1 = 1 / (1 + math.exp(-g1 * (r1 - r2) / 400))
        E2 = 1 / (1 + math.exp(-g2 * (r2 - r1) / 400))
        
        # Outcome (winner wins, loser loses)
        score1 = 1
        score2 = 0

        # Update ratings
        d1 = (Q / (1 / (rd1**2) + 1 / (g2**2 * E1 * (1 - E1))))
        d2 = (Q / (1 / (rd2**2) + 1 / (g1**2 * E2 * (1 - E2))))
        
        # Calculate new rating
        self.glicko_ratings[winner] = r1 + g1 * (score1 - E1) * K
        self.glicko_ratings[loser] = r2 + g2 * (score2 - E2) * K
        
        # Update RD after match
        self.glicko_rd[winner] = max(30, rd1 * decay)
        self.glicko_rd[loser] = max(30, rd2 * decay)
        
        # Optional: Return the updated ratings and RD
        return self.glicko_ratings[winner], self.glicko_ratings[loser], self.glicko_rd[winner], self.glicko_rd[loser]

    def expected_outcome(self, player1, player2, h_weight=12):
        """
        Calculate the expected outcome of a match between two players based on their Glicko ratings and RDs.

        Parameters:
            player1 (str): The name of the first player.
            player2 (str): The name of the second player.

        Returns:
            tuple: A tuple containing the expected probabilities for player1 and player2.
        """
        # Constants
        Q = math.log(10) / 400  # Glicko scaling constant

        # Retrieve Glicko ratings and RDs
        r1 = self.glicko_ratings[player1]
        rd1 = self.glicko_rd[player1]
        r2 = self.glicko_ratings[player2]
        rd2 = self.glicko_rd[player2]

        # Calculate g(RD) for each player
        g1 = 1 / math.sqrt(1 + (3 * Q**2 * rd1**2) / (math.pi**2))
        g2 = 1 / math.sqrt(1 + (3 * Q**2 * rd2**2) / (math.pi**2))

        # Calculate expected probabilities
        expected_probA = 1 / (1 + math.exp(-g1 * (r1 - r2) / 400))
        expected_probA_2 = 1 / (1 + math.exp(-g2 * (r2 - r1) / 400))

        # Adjust for head-to-head records
        if (player1, player2) in self.head_to_head:
            h2h_record = self.head_to_head[(player1, player2)]
            h1 = 1 + (h2h_record['wins'] - h2h_record['losses'])
            if h1 > 0:
                r1 += math.sqrt(h1) * h_weight
            else:
                r1 -= math.sqrt(abs(h1)) * h_weight
        if (player2, player1) in self.head_to_head:
            h2h_record = self.head_to_head[(player2, player1)]
            h2 = 1 + (h2h_record['wins'] - h2h_record['losses'])
            if h2 > 0:
                r2 += math.sqrt(h2) * h_weight  
            else:
                r2 -= math.sqrt(abs(h2)) * h_weight  

        # Calculate expected probabilities considering head-to-head adjustments
        expected_probH = 1 / (1 + math.exp(-g1 * (r1 - r2) / 400))
        expected_probH_2 = 1 / (1 + math.exp(-g2 * (r2 - r1) / 400))
        
        return expected_probA, expected_probA_2, None, None, expected_probH, expected_probH_2

    def update_head_to_head(self, winner, loser):
        """
        Update the head-to-head record between two players after a match.

        Parameters:
        winner (str): Name of the winning player.
        loser (str): Name of the losing player.

        Updates the wins and losses between the players in the head-to-head dictionary.
        """
        self.head_to_head.setdefault((winner, loser), {'wins': 0, 'losses': 0})
        self.head_to_head.setdefault((loser, winner), {'wins': 0, 'losses': 0})

        self.head_to_head[(winner, loser)]['wins'] += 1
        self.head_to_head[(loser, winner)]['losses'] += 1    

    def simulate_match(self, player1, player2, player1_wins, match_date, K=16, decay=0.8):
        """
        Simulate a match between two players, update Glicko ratings, and adjust head-to-head records.

        Parameters:
        player1 (str): Name of the first player.
        player2 (str): Name of the second player.
        player1_wins (bool): True if player 1 wins, False if player 2 wins.

        Updates Glicko ratings and head-to-head records based on the match outcome.
        """
        if player1_wins:
            self.update_glicko(player1, player2, match_date=match_date, K=K, decay=decay)
            self.update_head_to_head(player1, player2)
        else:
            self.update_glicko(player2, player1, match_date=match_date, K=K, decay=decay)
            self.update_head_to_head(player2, player1)        

    def calculate_and_analyze_glicko(self, K=12, H=10, decay=0.9, c=60, DELTA=timedelta(weeks=3)):
        """
        Initialize Glicko r and rd dictionaries, simulate matches,
        update dictionaries, and analyze results.
        
        Returns:
        Glicko dataframe
        """
        # Create a copy of matches to avoid modifying the original DataFrame
        self.matches_glicko = self.matches.copy()
        
        self.initialize_players()
        
        # Initialize columns for Glicko ratings and expected probabilities
        self.matches_glicko['r_winner_before'] = 0
        self.matches_glicko['r_winner_after'] = 0
        self.matches_glicko['rd_winner_before'] = 0
        self.matches_glicko['rd_winner_after'] = 0
        self.matches_glicko['expected_probA'] = 0.0
        
        # Initialize lists for storing actual results and predicted probabilities
        actual_results = []
        predicted_probs = []

        # Initialize dictionaries to store the last match date and ranking
        last_played = {}
        most_recent_ranking = {}
        match_count = {}
        matches_won = {}

        for index, row in self.matches_glicko.iterrows():
            player1 = row['Winner']
            player2 = row['Loser']
            match_date = row['Date']
            ranking_winner = row['WRank']
            ranking_loser = row['LRank']
            
            # Ensure both players have a Glicko rating
            if player1 not in self.glicko_ratings:
                self.glicko_ratings[player1] = 1500
            if player2 not in self.glicko_ratings:
                self.glicko_ratings[player2] = 1500
            
            # Update inactivity for both players before running calculation
            if player1 in last_played:
                self.update_rd_for_inactivity(player1, last_played[player1], match_date, c=c, DELTA=DELTA)
            if player2 in last_played:
                self.update_rd_for_inactivity(player2, last_played[player2], match_date, c=c, DELTA=DELTA)
            
            # Store current ratings before the match
            r_winner_before = round(self.glicko_ratings[player1], 1)
            r_loser_before = round(self.glicko_ratings[player2], 1)
            rd_winner_before = round(self.glicko_rd[player1], 1)
            rd_loser_before = round(self.glicko_rd[player2], 1)

            self.matches_glicko.loc[index, 'r_winner_before'] = r_winner_before
            self.matches_glicko.loc[index, 'rd_winner_before'] = rd_winner_before
            self.matches_glicko.loc[index, 'r_loser_before'] = r_loser_before       
            self.matches_glicko.loc[index, 'rd_loser_before'] = rd_loser_before
            
            # Calculate expected probabilities
            expected_probA, expected_probA_2, expected_probS, expected_probS_2, expected_probH, expected_probH_2 = self.expected_outcome(player1, player2, h_weight=H)
            self.matches_glicko.loc[index, 'expected_probA'] = expected_probA
            self.matches_glicko.loc[index, 'expected_probS'] = expected_probS
            self.matches_glicko.loc[index, 'expected_probH'] = expected_probH
            self.matches_glicko.loc[index, 'expected_probA_2'] = expected_probA_2
            self.matches_glicko.loc[index, 'expected_probS_2'] = expected_probS_2
            self.matches_glicko.loc[index, 'expected_probH_2'] = expected_probH_2
            
            
            # Determine who won and simulate the match
            player1_wins = row['Winner'] == player1
            self.simulate_match(player1, player2, player1_wins, match_date=match_date, K=K, decay=decay)
            
            # Store new ratings after the match
            r_winner_after = round(self.glicko_ratings[player1], 1)
            r_loser_after = round(self.glicko_ratings[player2], 1)
            rd_winner_after = round(self.glicko_rd[player1], 1)
            rd_loser_after = round(self.glicko_rd[player2], 1)

            self.matches_glicko.loc[index, 'r_winner_after'] = r_winner_after
            self.matches_glicko.loc[index, 'rd_winner_after'] = rd_winner_after
            self.matches_glicko.loc[index, 'r_loser_after'] = r_loser_after    
            self.matches_glicko.loc[index, 'rd_loser_after'] = rd_loser_after

            # Update the most recent match date and ranking for both players
            last_played[player1] = match_date
            most_recent_ranking[player1] = ranking_winner
            last_played[player2] = match_date
            most_recent_ranking[player2] = ranking_loser
            
            # Update match count and matches won for both players
            self.match_count[player1] = self.match_count.get(player1, 0) + 1
            self.match_count[player2] = self.match_count.get(player2, 0) + 1
            self.matches_won[player1] = self.matches_won.get(player1, 0) + 1 if player1_wins else self.matches_won.get(player1, 0)

            # Append actual result and predicted probability
            actual_results.append(1 if player1_wins else 0)
            predicted_probs.append(expected_probA)

        # Create DataFrames from the dictionaries
        last_played_df = pd.DataFrame.from_dict(last_played, orient='index', columns=['Last_Match_Date'])
        recent_ranking_df = pd.DataFrame.from_dict(most_recent_ranking, orient='index', columns=['Most_Recent_Ranking'])
        match_count_df = pd.DataFrame.from_dict(self.match_count, orient='index', columns=['Match_Count'])
        matches_won_df = pd.DataFrame.from_dict(self.matches_won, orient='index', columns=['Matches_Won'])

        # Calculate win percentage and add it to the matches_won_df DataFrame
        matches_won_df['Win_Percentage'] = matches_won_df['Matches_Won'] / match_count_df['Match_Count'] * 100
        
        # Combine all Glicko ratings into a single DataFrame
        glicko_r_df = pd.DataFrame.from_dict(self.glicko_ratings, orient='index', columns=['R_value'])
        glicko_rd_df = pd.DataFrame.from_dict(self.glicko_rd, orient='index', columns=['RD_value'])
        
        # Merge the dataframes to create a consolidated view 
        glicko_combined_df = glicko_r_df.join([glicko_rd_df, last_played_df,
                                               recent_ranking_df, match_count_df, 
                                               matches_won_df])

        # Sort the dataframe by the last match date
        glicko_combined_df.sort_values(by='R_value', ascending=False, inplace=True)
        self.glicko_df = glicko_combined_df.fillna(0) 
        
        # Calculate accuracy and log loss
        correct_predictions = sum(1 for prob in predicted_probs if prob >= 0.5)
        accuracy = correct_predictions / len(predicted_probs) * 100
        
        predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)  # Avoid log(0) errors
        log_loss_value = -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for y, p in zip(actual_results, predicted_probs)])
        
        print("Glicko Calculations completed...")
        print(f"Accuracy of Baseline Model: {accuracy:.2f}%")
        print(f"Log Loss of Baseline Model: {log_loss_value:.4f}")
        
        return self.glicko_df, self.matches_glicko


# %% debugging glicko class
"""
# Initialize the class with players and matches DataFrames
glicko_functions = GlickoFunctions(players=players, matches=matches, K=23, c=60, decay=0.875)

# Run the Elo calculation
glicko_df, matches_glicko = glicko_functions.calculate_and_analyze_glicko(K=12, H=20, decay=0.85, c=60, DELTA=timedelta(weeks=3))
print(glicko_df.head(5))
# Later, if you want to access the Elo and matches DataFrames again:
#elo_df, matches_elo = elo_functions.get_elo_data()        
"""            
