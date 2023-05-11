#%% imports

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

raw_data = pd.read_csv('bootbag_playerround.csv')
raw_data = raw_data[raw_data['roundId'].notna()]

#test = raw_data[raw_data['playerId'] == player_id]

raw_data['createdAt'] = pd.to_datetime(raw_data['createdAt'])
raw_data = raw_data.sort_values(by='createdAt', ascending=True)
raw_data = raw_data[raw_data["createdAt"] >= "2022-01-01"]

raw_data = raw_data.dropna(axis=0)

#%% Looping through all rounds to get maximum potential scores

all_rounds_players = []
for x in raw_data['roundId']:
    if x not in all_rounds_players:
        all_rounds_players.append(x)

all_round_changes_wide = pd.DataFrame()
all_round_changes_long = pd.DataFrame()

game_round = '76KyvqPeqnlJb3dffB3U'
i=0
for i, game_round in enumerate(tqdm(all_rounds_players, position=0, leave=True)):
    round_number = i+1
    round_data = raw_data[raw_data['roundId'] == game_round]
    
    all_player_rating_change_round = pd.DataFrame()
    
    player_id = '6z1du1o69dgamvu2m8yi97dre'
    for player_id in list(set(round_data['playerId'])):

        player_round_data = round_data[round_data['playerId'] == player_id]
        player_round_data = player_round_data.reset_index(drop=True)
        
        player_round_score_multiplier = player_round_data['scoreMultiplier'].iloc[0]
        
        if math.isnan(player_round_data['prevRating'].iloc[0]):
            start_rating = player_round_data['rating'].iloc[0]
        else:
            start_rating = player_round_data['prevRating'].iloc[0]
        
        #need to add start rating to possibe buy-ins
        all_player_values = [start_rating]
        all_player_values.extend(player_round_data['rating'])
        
        player_rating_array = np.array(all_player_values)
        #get upper-right triangle of subtraction matrix (only looks forward, not back like it would in bottom-left)
        player_rating_array = -np.triu(player_rating_array[:,None] - player_rating_array)
        
        player_rating_array_column = player_rating_array[np.triu_indices(n=len(player_rating_array),k=1)]
        
        #maximum value will be greatest possible change
        best_playerround_ratingdiff = player_rating_array_column.max()
        
        buy_in_index = np.where(player_rating_array == best_playerround_ratingdiff)[0][0]
        buy_in_rating = all_player_values[buy_in_index]
        
        sell_out_index = np.where(player_rating_array == best_playerround_ratingdiff)[1][0]
        sell_out_rating = all_player_values[sell_out_index]
        
        player_round_bestscore = (((sell_out_rating - buy_in_rating) / buy_in_rating) * 100) * player_round_score_multiplier
        
        player_rating_change_round = pd.DataFrame([[player_id, player_round_bestscore]], columns=['Player ID', f'{game_round}'])
        
        all_player_rating_change_round = pd.concat([all_player_rating_change_round, player_rating_change_round])

    ax = sns.histplot(all_player_rating_change_round[f'{game_round}'],
                      bins=20,
                      kde=True,
                      color='green')
    ax.set(xlabel=f'Round {round_number}', ylabel='Frequency')
    
    plt.show()
    
    all_player_rating_change_round.index = all_player_rating_change_round['Player ID']
    all_player_rating_change_round = all_player_rating_change_round.drop('Player ID', axis=1)
    
    all_round_changes_wide = pd.concat([all_round_changes_wide, all_player_rating_change_round], axis=1)

    all_player_rating_change_round.columns = ['player_scores']
    
    all_round_changes_long = pd.concat([all_round_changes_long, all_player_rating_change_round.reset_index(drop=True)], axis=0)
    
    
all_round_changes_long = all_round_changes_long.reset_index(drop=True).sort_values(by='player_scores', ascending=False).reset_index(drop=True)

all_round_changes_wide = all_round_changes_wide.reset_index()
all_round_changes_wide.to_excel('all_rounds_final_scores.xlsx', index=False)


#%% all players combined scores all rounds

sns.set(rc={'figure.figsize':(11.7,8.27)}, style='white')

ax = sns.histplot(all_round_changes_long['player_scores']*25,
                  bins=40,
                  kde=True,
                  color='green')
ax.set(xlabel=f'Player Scores (All Rounds)', ylabel='Number of Players')

plt.show()

(all_round_changes_long['player_scores']*25).describe()


#%% top 20s

top_20_perround = pd.DataFrame()
for column in all_round_changes_wide:
    if column != 'Player ID':
        top_20_perround = pd.concat([top_20_perround, pd.DataFrame(all_round_changes_wide.sort_values(by=column, ascending=False)[column].reset_index(drop=True)[:20])], axis=1)
        
top_20_sums = top_20_perround.sum(axis=0).reset_index()
top_20_sums.columns = ['Round Number', 'Best Possible Score']

#%% Transactions & Points Actually Obtained

transactions_raw = pd.read_csv('bootbag_transactionround.csv', dtype=str)

transactions_raw = transactions_raw[transactions_raw['type'] == 'sell'].dropna(subset=['points'])
transactions_raw['points'] = transactions_raw['points'].astype(int)
transactions_raw['coins'] = transactions_raw['coins'].astype(float)

#%% Transaction Coin Breakdown

new_times = []
for time in transactions_raw['timestamp']:
    try:
        new_time = datetime.fromtimestamp(int(time)/1000)
    except:
        new_time = None
    new_times.append(new_time)
    
transactions_raw['timestamp'] = new_times

all_rounds_transactions = []
for x in transactions_raw['roundId']:
    if x not in all_rounds_transactions:
        all_rounds_transactions.append(x)

transactions_raw = transactions_raw[~transactions_raw.roundId.isin(all_rounds_players)]
len(list(set(transactions_raw['roundId'])))

game_round = 'hJlPZZ68ldg1KmMtx4gM'

all_user_scores = pd.DataFrame()
for game_round in list(set(transactions_raw['roundId'])):
    round_df = transactions_raw[transactions_raw['roundId'] == game_round]
    round_user_scores = []
    for user_id in list(set(round_df['userId'])):
        user_round_df = round_df[round_df['userId'] == user_id]
        user_round_score = user_round_df['points'].sum()
        round_user_scores.append([user_round_score, game_round])
        
        
    round_user_scores = pd.DataFrame(round_user_scores, columns = ['user_score', 'roundId'])
    
    ax = sns.histplot(round_user_scores['user_score'],
                      bins=5,
                      kde=True,
                      color='green')
    ax.set(xlabel=game_round, ylabel='Frequency')

    plt.show()
    
    print(len(round_user_scores))
    
    all_user_scores = pd.concat([all_user_scores, round_user_scores], axis=0)


ax = sns.histplot(all_user_scores['user_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()

all_user_scores['user_score'].describe()

#%%




    

#%% Permutations - 25 Coins on 20 Random Top 300 Player Rounds (All Time)

top_300_players = all_round_changes_long[:300].copy()
top_300_players['coins'] = 25
top_300_players['user_score'] = top_300_players['player_scores'] * top_300_players['coins']

top_300_players = np.array(top_300_players['user_score'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    selected_players = np.random.choice(top_300_players, size=20)
    user_round_score = selected_players.sum()
    user_round_scores.append(user_round_score)

user_round_scores_25 = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_25['user_round_score'].iloc[0])
print(user_round_scores_25['user_round_score'].mean())
print(user_round_scores_25['user_round_score'].iloc[20])


ax = sns.histplot(user_round_scores_25['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_25.pkl', 'wb') as f:
    pickle.dump(user_round_scores_25, f)

#%%

with open('user_round_scores_25.pkl', 'rb') as f:
    user_round_scores_25 = pickle.load(f)
    
#%% Permutations - Random(ish) Coins Across Random Sample of Top 300 Players

import random

top_300_players = all_round_changes_long[:300].copy()

random_players = np.array(top_300_players['player_scores'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    coins_split = [25,25,25,25,25]
    total_coins = 125
    while total_coins < 500:
        coins_bet = random.randint(1,25)
        if total_coins + coins_bet > 500:
            coins_bet = coins_bet - ((coins_bet + total_coins) - 500)
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
        else:
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
    
    selected_players = np.random.choice(random_players, size=len(coins_split))
    user_scores = selected_players*coins_split
    user_round_score = user_scores.sum()
    user_round_scores.append(user_round_score)

user_round_scores_rand = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_rand['user_round_score'].iloc[0])
print(user_round_scores_rand['user_round_score'].mean())
print(user_round_scores_rand['user_round_score'].iloc[20])

ax = sns.histplot(user_round_scores_rand['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_rand.pkl', 'wb') as f:
    pickle.dump(user_round_scores_rand, f)

#%% 

with open('user_round_scores_rand.pkl', 'rb') as f:
    user_round_scores_rand = pickle.load(f)

#%% Permutations - Coin Spread Like Real Life

coins_poss = [25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
coin_probabilities = [0.27351,0.00417,0.00275,0.00285,0.00377,0.04814,0.00440,0.00560,0.00599,0.00791,0.07032,0.01076,0.01169,0.01324,0.01655,0.15230,0.01162,0.01851,0.01536,0.01881,0.09234,0.01586,0.01622,0.01020,0.16713]


#PICK RANDOM TOP 50 PLAYER SCORES AND ASSIGN A COIN FROM LIST ABOVE, MULTIPLY SCORE AND SUM. REPEAT 000's of times then get 1/300,000th chance
top_300_players = all_round_changes_long[:300].copy()
top_300_players = np.array(top_300_players['player_scores'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    coins_split = []
    total_coins = 0
    while total_coins < 500:
        coins_bet = np.random.choice(coins_poss, 1, p=coin_probabilities)[0]
        if total_coins + coins_bet > 500:
            coins_bet = coins_bet - ((coins_bet + total_coins) - 500)
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
        else:
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
            
    selected_players = np.random.choice(top_300_players, size=len(coins_split))
    user_scores = selected_players*coins_split

    user_round_score = user_scores.sum()
    user_round_scores.append(user_round_score)


user_round_scores_dist = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_dist['user_round_score'].iloc[0])
print(user_round_scores_dist['user_round_score'].mean())
print(user_round_scores_dist['user_round_score'].iloc[20])



ax = sns.histplot(user_round_scores_dist['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_dist.pkl', 'wb') as f:
    pickle.dump(user_round_scores_dist, f)



with open('user_round_scores_dist.pkl', 'rb') as f:
    user_round_scores_dist = pickle.load(f)


#%% Same as above, with Active Players Only

active_players = all_round_changes_wide.dropna(thresh=11)
active_players.index = active_players['Player ID']
active_players = active_players.drop('Player ID', axis=1)

all_active_scores = []
for round_id in active_players:
    active_round_df = active_players[round_id]
    all_active_scores.extend(list(active_round_df))

active_players_long = pd.DataFrame(all_active_scores, columns=['player_scores']).sort_values(by='player_scores', ascending=False).dropna()


#%% Permutations - 25 Coins on 20 Random Active Player Scores

random_players = active_players_long.copy()
random_players['coins'] = 25
random_players['user_score'] = random_players['player_scores'] * random_players['coins']

random_players = np.array(random_players['user_score'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    selected_players = np.random.choice(random_players, size=20)
    user_round_score = selected_players.sum()
    user_round_scores.append(user_round_score)

user_round_scores_25_active = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_25_active['user_round_score'].iloc[0])
print(user_round_scores_25_active['user_round_score'].mean())
print(user_round_scores_25_active['user_round_score'].iloc[20])


ax = sns.histplot(user_round_scores_25_active['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_25_active.pkl', 'wb') as f:
    pickle.dump(user_round_scores_25_active, f)


#%%

with open('user_round_scores_25_active.pkl', 'rb') as f:
    user_round_scores_25_active = pickle.load(f)


#%% Permutations - Random(ish) Coins Across Random Sample of Active Players

import random

random_players = active_players_long.copy()
random_players = np.array(random_players['player_scores'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    coins_split = [25,25,25,25,25]
    total_coins = 125
    while total_coins < 500:
        coins_bet = random.randint(1,25)
        if total_coins + coins_bet > 500:
            coins_bet = coins_bet - ((coins_bet + total_coins) - 500)
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
        else:
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
    
    selected_players = np.random.choice(random_players, size=len(coins_split))
    user_scores = selected_players*coins_split

    user_round_score = user_scores.sum()
    user_round_scores.append(user_round_score)

user_round_scores_rand_active = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_rand_active['user_round_score'].iloc[0])
print(user_round_scores_rand_active['user_round_score'].mean())
print(user_round_scores_rand_active['user_round_score'].iloc[20])


ax = sns.histplot(user_round_scores_rand_active['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_rand_active.pkl', 'wb') as f:
    pickle.dump(user_round_scores_rand_active, f)

#%% 

with open('user_round_scores_rand_active.pkl', 'rb') as f:
    user_round_scores_rand_active = pickle.load(f)
    
    
#%% Permutations - Coin Spread Like Real Life

coins_poss = [25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
coin_probabilities = [0.27351,0.00417,0.00275,0.00285,0.00377,0.04814,0.00440,0.00560,0.00599,0.00791,0.07032,0.01076,0.01169,0.01324,0.01655,0.15230,0.01162,0.01851,0.01536,0.01881,0.09234,0.01586,0.01622,0.01020,0.16713]


#PICK RANDOM TOP 50 PLAYER SCORES AND ASSIGN A COIN FROM LIST ABOVE, MULTIPLY SCORE AND SUM. REPEAT 000's of times then get 1/300,000th chance
random_players = active_players_long.copy()
random_players = np.array(random_players['player_scores'])

user_round_scores = []
for n in tqdm(range(6000000),position=0, leave=True):
    coins_split = []
    total_coins = 0
    while total_coins < 500:
        coins_bet = np.random.choice(coins_poss, 1, p=coin_probabilities)[0]
        if total_coins + coins_bet > 500:
            coins_bet = coins_bet - ((coins_bet + total_coins) - 500)
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
        else:
            coins_split.append(coins_bet)
            total_coins = total_coins + coins_bet
            
    selected_players = np.random.choice(random_players, size=len(coins_split))
    user_scores = selected_players*coins_split

    user_round_score = user_scores.sum()
    user_round_scores.append(user_round_score)


user_round_scores_dist_active = pd.DataFrame(user_round_scores, columns=['user_round_score']).sort_values(by='user_round_score', ascending=False).reset_index(drop=True)
print(user_round_scores_dist_active['user_round_score'].iloc[0])
print(user_round_scores_dist_active['user_round_score'].mean())
print(user_round_scores_dist_active['user_round_score'].iloc[20])


ax = sns.histplot(user_round_scores_dist_active['user_round_score'],
                  bins=30,
                  kde=True,
                  color='green')
ax.set(xlabel="User Score", ylabel='Number of Times Obtained')

plt.show()


with open('user_round_scores_dist_active.pkl', 'wb') as f:
    pickle.dump(user_round_scores_dist_active, f)
    
    
#%%

with open('user_round_scores_dist_active.pkl', 'rb') as f:
    user_round_scores_dist_active = pickle.load(f)
    


