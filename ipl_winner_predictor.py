import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Load the JSON data
with open('ipl_history_2008_2026.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Clean column names (remove trailing spaces)
df.columns = df.columns.str.strip()

# Also clean string values in the dataframe
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Feature Engineering
def create_features(df):
    features = []
    
    # Team performance history
    team_wins = Counter()
    team_runner_up = Counter()
    captain_wins = Counter()
    venue_wins = Counter()
    
    for idx, row in df.iterrows():
        season = row['Season']  # ✅ Fixed: Removed trailing space
        
        # Features for winner
        winner = row['Winner']  # ✅ Fixed
        winner_captain = row['Winner_Captain']  # ✅ Fixed
        runner_up = row['Runner_Up']  # ✅ Fixed
        venue = row['Final_Venue']  # ✅ Fixed
        
        # Historical wins before this season
        winner_prev_wins = team_wins[winner]
        runner_up_prev_wins = team_wins[runner_up]
        
        # Captain's previous wins
        captain_prev_wins = captain_wins[winner_captain]
        
        # Venue performance
        venue_winner_wins = venue_wins.get(f"{winner}_{venue}", 0)
        
        # Recent form (last 3 seasons)
        recent_seasons = df[df['Season'] < season].tail(3)
        winner_recent = (recent_seasons['Winner'] == winner).sum()
        
        feature = {
            'season': season,
            'winner': winner,
            'runner_up': runner_up,
            'winner_captain': winner_captain,
            'venue': venue,
            'winner_prev_wins': winner_prev_wins,
            'runner_up_prev_wins': runner_up_prev_wins,
            'captain_prev_wins': captain_prev_wins,
            'venue_winner_wins': venue_winner_wins,
            'winner_recent_form': winner_recent,
            'is_defending_champion': 1 if winner_prev_wins > 0 else 0
        }
        
        features.append(feature)
        
        # Update counters
        team_wins[winner] += 1
        team_runner_up[runner_up] += 1
        captain_wins[winner_captain] += 1
        venue_wins[f"{winner}_{venue}"] = venue_wins.get(f"{winner}_{venue}", 0) + 1
    
    return pd.DataFrame(features), team_wins

# Create features
features_df, team_wins = create_features(df)

# Encode categorical variables
le_winner = LabelEncoder()
le_runner_up = LabelEncoder()
le_captain = LabelEncoder()
le_venue = LabelEncoder()

# Fit on all unique values
all_teams = list(set(features_df['winner'].tolist() + features_df['runner_up'].tolist()))
le_winner.fit(all_teams)
le_runner_up.fit(all_teams)

all_captains = features_df['winner_captain'].unique()
le_captain.fit(all_captains)

all_venues = features_df['venue'].unique()
le_venue.fit(all_venues)

# Prepare training data
X = []
y = []

for idx, row in features_df.iterrows():
    # Winner encoded
    X.append([
        row['winner_prev_wins'],
        row['runner_up_prev_wins'],
        row['captain_prev_wins'],
        row['winner_recent_form'],
        row['is_defending_champion'],
        le_winner.transform([row['winner']])[0],
        le_runner_up.transform([row['runner_up']])[0],
        le_captain.transform([row['winner_captain']])[0],
        le_venue.transform([row['venue']])[0]
    ])
    y.append(1)  # Winner class
    
    # Runner-up (negative example)
    X.append([
        row['runner_up_prev_wins'],
        row['winner_prev_wins'],
        row['captain_prev_wins'],
        0,  # runner-up recent form (not winner)
        0,  # not defending champion
        le_winner.transform([row['runner_up']])[0],
        le_runner_up.transform([row['winner']])[0],
        le_captain.transform([row['winner_captain']])[0],
        le_venue.transform([row['venue']])[0]
    ])
    y.append(0)  # Loser class

X = np.array(X)
y = np.array(y)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X, y)

# Save model and encoders
joblib.dump({
    'model': model,
    'le_winner': le_winner,
    'le_runner_up': le_runner_up,
    'le_captain': le_captain,
    'le_venue': le_venue,
    'team_wins': dict(team_wins),
    'feature_names': [
        'team_prev_wins',
        'opponent_prev_wins',
        'captain_prev_wins',
        'recent_form',
        'is_defending_champion',
        'team_encoded',
        'opponent_encoded',
        'captain_encoded',
        'venue_encoded'
    ]
}, 'ipl_model.pkl')

print("✅ IPL Winner Predictor model trained and saved as 'ipl_model.pkl'")
print(f"📊 Model accuracy on training data: {model.score(X, y):.2%}")
print(f"📈 Total seasons analyzed: {len(features_df)}")
print(f"🏆 Teams in dataset: {len(all_teams)}")