#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:05:04 2024

@author: kaalvoetranger@gmail.com

ABOUT: These are mainly helper functions for main.py that don't need data caching

"""

#%% dependencies

import pandas as pd
import requests
import numpy as np
import os
import time
import plotly.graph_objs as go
import streamlit as st

#%% local variables (these are generated in main.py)
app_dir = os.getcwd()
data_dir = os.path.expanduser("~app_dir/datasets/")
g_url1 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/matches.csv'
g_url2 = 'https://raw.githubusercontent.com/kaalvoetranger-88/st-habibies-bets/main/datasets/atp_players.csv'
matches = pd.read_csv(g_url1)
players = pd.read_csv(g_url2)
    

#%%         functions for elo calculations


def initialize_elo_ratings():
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
    return elo_dict.setdefault(player, 'Value')


def update_elo(winner, loser, elo_dict, K=32):
    winner_elo = get_elo(winner, elo_dict)
    loser_elo = get_elo(loser, elo_dict)
    
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    
    elo_dict[winner] = winner_elo + K * (1 - expected_winner)
    elo_dict[loser] = loser_elo + K * (0 - expected_loser)


def expected_outcome(player1, player2, surface, weight_surface=0.9, h2h_weight=10):
    weight_all = 1 - weight_surface
    
    elo1_all = elo_ratings_all[player1]
    elo2_all = elo_ratings_all[player2]
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
            
    combined_elo1 = weight_all * elo1_all + weight_surface * elo1_surface
    combined_elo2 = weight_all * elo2_all + weight_surface * elo2_surface
    
    expected_probA = 1 / (1 + 10 ** ((elo2_all - elo1_all) / 400))
    expected_probS = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
    
    if (player1, player2) in head_to_head:
        h2h_record = head_to_head[(player1, player2)]
        player1_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
        combined_elo1 += h2h_weight * player1_h2h_advantage
    
    if (player2, player1) in head_to_head:
        h2h_record = head_to_head[(player2, player1)]
        player2_h2h_advantage = h2h_record['wins'] - h2h_record['losses']
        combined_elo2 += h2h_weight * player2_h2h_advantage
    
    expected_probH = 1 / (1 + 10 ** ((combined_elo2 - combined_elo1) / 400))
    
    return expected_probA, expected_probS, expected_probH



def update_head_to_head(winner, loser):
    head_to_head.setdefault((winner, loser), {'wins': 0, 'losses': 0})
    head_to_head.setdefault((loser, winner), {'wins': 0, 'losses': 0})
    
    head_to_head[(winner, loser)]['wins'] += 1
    head_to_head[(loser, winner)]['losses'] += 1


def get_rating(player):
    return elo_ratings_all.get(player, 'Value')


def simulate_match(player1, player2, player1_wins, surface):
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
    global elo_df
    initialize_elo_ratings()
    
    # Initialize columns for Elo and expected probability
    matches['elo_winner_before'] = 0
    matches['elo_winner_after'] = 0
    matches['expected_probA'] = 0.0
    matches['expected_probS'] = 0.0
    matches['expected_probH'] = 0.0
    matches['elo_loser_before'] = 0
    matches['elo_loser_after'] = 0

    # Initialize lists for storing actual results and predicted probabilities
    actual_results = []
    predicted_probs = []

    for index, row in matches.iterrows():
        player1 = row['winner_name']
        player2 = row['loser_name']
        surface = row['surface']
        match_date = row['tourney_date']
        
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
        
        # Determine who won and simulate the match
        player1_wins = row['winner_name'] == player1
        simulate_match(player1, player2, player1_wins, surface)
        
        # Update the Elo ratings after the match
        elo_winner_after = round(elo_ratings_all[player1], 1)
        elo_loser_after = round(elo_ratings_all[player2], 1)
        matches.loc[index, 'elo_winner_after'] = elo_winner_after
        matches.loc[index, 'elo_loser_after'] = elo_loser_after

        # Update the most recent match date for both players
        most_recent_date[player1] = max(most_recent_date.get(player1, match_date), match_date)
        most_recent_date[player2] = max(most_recent_date.get(player2, match_date), match_date)
        
        # Update match count and matches won for both players
        match_count[player1] = match_count.get(player1, 0) + 1
        match_count[player2] = match_count.get(player2, 0) + 1
        matches_won[player1] = matches_won.get(player1, 0) + 1

        # Append actual result and predicted probability
        actual_results.append(1 if player1_wins else 0)
        predicted_probs.append(expected_probH)

    # Create DataFrames from the dictionaries
    recent_date_df = pd.DataFrame.from_dict(most_recent_date, orient='index', columns=['Most_Recent_Date'])
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
    elo_combined_df = elo_all_df.join([elo_grass_df, elo_clay_df, elo_hard_df, recent_date_df, match_count_df, matches_won_df])

    # Sort the dataframe by overall Elo rating (optional)
    elo_combined_df.sort_values(by='Elo_ALL', ascending=False, inplace=True)
    elo_df = elo_combined_df.copy()
    elo_df['Elo_ALL'] = elo_df['Elo_ALL'].round(0).astype(int)
    elo_df['Elo_Hard'] = elo_df['Elo_Hard'].round(0).astype(int)
    elo_df['Elo_Clay'] = elo_df['Elo_Clay'].round(0).astype(int)
    elo_df['Elo_Grass'] = elo_df['Elo_Grass'].round(0).astype(int)
    
    # Calculate accuracy and log loss
    correct_predictions = sum(1 for prob in predicted_probs if prob >= 0.5)
    accuracy = correct_predictions / len(predicted_probs) * 100
    
    # Manually calculate log loss
    predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)  # Avoid log(0) errors
    log_loss_value = -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for y, p in zip(actual_results, predicted_probs)])
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Log Loss: {log_loss_value:.4f}")
    return elo_df, matches


#%%         functions for player info



def player_info_tool(players, matches, elo_df):
    # Search bar to search for a player by name
    player_name_input = st.text_input("Search for a player by name", "")

    if player_name_input:
        # Ensure 'Player Name' column has no NaN values for filtering
        players['Player Name'] = players['Player Name'].fillna("")
        
        # Filter the players DataFrame by player name
        filtered_players = players[players['Player Name'].str.contains(player_name_input, case=False, na="")]

        if not filtered_players.empty:
            print("Dropped Player knowledge on someone...")
            for _, row in filtered_players.iterrows():
                # Display player details from the `players` DataFrame
                st.subheader(f"Player: {row['Player Name']}")
                
                # Use columns to split the layout into three parts
                col1, col2, col3 = st.columns(3)
                
                # Left column for player name and DOB
                with col1:
                    if pd.notna(row['dob']):
                        st.write(f"Date of Birth: {row['dob'].strftime('%Y-%m-%d')}")
                    else:
                        st.write("Date of Birth: Not available")
                
                # Second column for Elo ratings without 'Elo_' prefix
                with col2:
                    # Check if the player exists in elo_df
                    if row['Player Name'] in elo_df.index:
                        elo_row = elo_df.loc[row['Player Name']]
                        
                        # Display Elo Ratings
                        st.write("Elo Ratings:")
                        for key, value in elo_row.items():
                            if key == "Elo_ALL":
                                st.write(f"Overall: {value}")
                            elif key == "Elo_Grass":
                                st.write(f"Grass: {value}")
                            elif key == "Elo_Clay":
                                st.write(f"Clay: {value}")
                            elif key == "Elo_Hard":
                                st.write(f"Hard: {value}")

                    else:
                        st.write("Elo ratings not available.")
                
                # Third column for Win_Percentage gauge
                with col3:
                    # Check if the player exists in elo_df
                    if row['Player Name'] in elo_df.index:
                        elo_row = elo_df.loc[row['Player Name']]
                        win_percentage = elo_row.get('Win_Percentage', 0)

                        # Create a gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=win_percentage,
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': 'blue'},
                                'bgcolor': 'orange',
                                'steps': [
                                    {'range': [0, 40], 'color': 'red'},
                                    {'range': [40, 70], 'color': 'orange',
                                     'range': [70, 100], 'color': 'green'}
                                ]
                            },
                            title={'text': "Win Percentage", 'font': {'size': 20}},
                            domain={'x': [0, 1], 'y': [0, 1]}
                        ))

                        st.plotly_chart(fig)
                    else:
                        st.write("Win percentage not available.")
                
                # Show recent match results from `matches`
                recent_matches = matches[
                    (matches['winner_name'] == row['Player Name']) | 
                    (matches['loser_name'] == row['Player Name'])
                ].copy()
                
                # Track player ranking and Elo ratings
                recent_matches['Player_Rank'] = recent_matches.apply(
                    lambda x: x['WRank'] if x['winner_name'] == row['Player Name'] else x['LRank'], axis=1
                )
                
                recent_matches['Player_Elo_After'] = recent_matches.apply(
                    lambda x: x['elo_winner_after'] if x['winner_name'] == row['Player Name'] else x['elo_loser_after'], axis=1
                )
                
                # Sort the matches by date for the plot
                recent_matches = recent_matches.sort_values(by='tourney_date')
                
                if not recent_matches.empty:
                    # Display recent matches table without the 'info' column
                    st.subheader("Recent Matches:")
                    st.write('20 most recent')
                    match_display = recent_matches.tail(20)
                    match_display = match_display.copy()
                    match_display['Player_Elo_After'] = match_display['Player_Elo_After'].round(0)

                    st.dataframe(match_display[['tourney_date', 'Tournament', 'surface', 
                                                'Round', 'winner_name', 'loser_name', 
                                                'Scores', 'Player_Rank', 'Player_Elo_After']], 
                                 hide_index=True)
                    
                    # Plot rankings and Elo rating over time using Plotly
                    fig = go.Figure()

                    # Add the ranking data to the plot
                    fig.add_trace(go.Scatter(
                        x=recent_matches['tourney_date'],
                        y=recent_matches['Player_Rank'],
                        mode='lines+markers',
                        name='Player Ranking',
                        line=dict(color='blue'),
                        yaxis='y1'
                    ))

                    # Add the Elo rating data to the plot
                    fig.add_trace(go.Scatter(
                        x=recent_matches['tourney_date'],
                        y=recent_matches['Player_Elo_After'],
                        mode='lines+markers',
                        name='Player Elo Rating',
                        line=dict(color='red'),
                        yaxis='y2'
                    ))

                    # Layout for dual axis
                    fig.update_layout(
                        title=f"{row['Player Name']} - Ranking and Elo Rating Over Time",
                        xaxis_title="Date",
                        yaxis=dict(
                            title='Player Ranking',
                            side='left',
                            showgrid=False,
                            zeroline=False,
                            autorange="reversed",  # Rankings are better when they are lower
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
                    st.write("No recent matches found.")
        else:
            st.write("No player found with that name.")
    else:
        st.write("Enter a player name to search.")

  



#%%         functions for player comparison



def calculate_age(dob):
    if isinstance(dob, np.datetime64):
        dob = pd.to_datetime(dob).to_pydatetime()
    elif isinstance(dob, str):
        dob = datetime.strptime(dob, '%Y-%m-%d')
    elif isinstance(dob, datetime):
        pass  # Already a datetime object
    else:
        raise TypeError("Date of birth must be a string, numpy.datetime64, or datetime object")

    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age


# Function to generate a horizontal bar chart for a single player's Elos
def plot_player_elo(player_info, player_name, position='left'):
    # Create a DataFrame from the player_info
    df = pd.DataFrame({
        'Stat': ['Overall Elo', 'Grass Elo', 'Clay Elo', 'Hard Elo'],
        'Value': [
            player_info['elo_all'],
            player_info['elo_grass'],
            player_info['elo_clay'],
            player_info['elo_hard'] 
                ]
    })

    # Define bar colors and x-axis position based on the player position
    bar_color = 'blue' if position == 'left' else 'red'
  
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
        title="Current Elo Ratings",
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
        showlegend=False)
    #fig.update_xaxes(range=[1000, 2500])
    return fig





#%%         functions for matchmaking


#%%         functions for odds converter/calculator


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





#%%         functions for tournament simulation



def t_simulate_match(player1, player2, elo_df, elo_column='Elo_ALL'):
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
    winners = []
    for i in range(0, len(draw), 2):
        player1 = draw[i]
        player2 = draw[i + 1]
        winner = t_simulate_match(player1, player2, elo_df, elo_column)
        winners.append(winner)
        #time.sleep(0.1) for debugging 
    return winners



def simulate_tournament(player_list, elo_df, elo_column='Elo_ALL'):
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






