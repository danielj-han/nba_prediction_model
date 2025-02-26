from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import datetime
from nba_api.stats.endpoints import teamgamelog, scoreboardv2

# Load the pre-trained model (pipeline with StandardScaler + XGBoost)
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Mapping of team names to NBA team IDs
TEAM_DICT = {
    'Atlanta Hawks': '1610612737',
    'Boston Celtics': '1610612738',
    'Brooklyn Nets': '1610612751',
    'Charlotte Hornets': '1610612766',
    'Chicago Bulls': '1610612741',
    'Cleveland Cavaliers': '1610612739',
    'Dallas Mavericks': '1610612742',
    'Denver Nuggets': '1610612743',
    'Detroit Pistons': '1610612765',
    'Golden State Warriors': '1610612744',
    'Houston Rockets': '1610612745',
    'Indiana Pacers': '1610612754',
    'Los Angeles Clippers': '1610612746',
    'Los Angeles Lakers': '1610612747',
    'Memphis Grizzlies': '1610612763',
    'Miami Heat': '1610612748',
    'Milwaukee Bucks': '1610612749',
    'Minnesota Timberwolves': '1610612750',
    'New Orleans Pelicans': '1610612740',
    'New York Knicks': '1610612752',
    'Oklahoma City Thunder': '1610612760',
    'Orlando Magic': '1610612753',
    'Philadelphia 76ers': '1610612755',
    'Phoenix Suns': '1610612756',
    'Portland Trail Blazers': '1610612757',
    'Sacramento Kings': '1610612758',
    'San Antonio Spurs': '1610612759',
    'Toronto Raptors': '1610612761',
    'Utah Jazz': '1610612762',
    'Washington Wizards': '1610612764'
}

# Mapping of team names to logo URLs
TEAM_LOGOS = {
    'Atlanta Hawks': 'https://upload.wikimedia.org/wikipedia/en/2/24/Atlanta_Hawks_logo.svg',
    'Boston Celtics': 'https://upload.wikimedia.org/wikipedia/en/8/8f/Boston_Celtics.svg',
    'Brooklyn Nets': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/40/Brooklyn_Nets_primary_icon_logo_2024.svg/1024px-Brooklyn_Nets_primary_icon_logo_2024.svg.png',
    'Charlotte Hornets': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c4/Charlotte_Hornets_%282014%29.svg/220px-Charlotte_Hornets_%282014%29.svg.png',
    'Chicago Bulls': 'https://upload.wikimedia.org/wikipedia/en/6/67/Chicago_Bulls_logo.svg',
    'Cleveland Cavaliers': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Cleveland_Cavaliers_logo.svg/170px-Cleveland_Cavaliers_logo.svg.png',
    'Dallas Mavericks': 'https://upload.wikimedia.org/wikipedia/en/9/97/Dallas_Mavericks_logo.svg',
    'Denver Nuggets': 'https://upload.wikimedia.org/wikipedia/en/7/76/Denver_Nuggets.svg',
    'Detroit Pistons': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Logo_of_the_Detroit_Pistons.svg/200px-Logo_of_the_Detroit_Pistons.svg.png',
    'Golden State Warriors': 'https://upload.wikimedia.org/wikipedia/en/0/01/Golden_State_Warriors_logo.svg',
    'Houston Rockets': 'https://upload.wikimedia.org/wikipedia/en/2/28/Houston_Rockets.svg',
    'Indiana Pacers': 'https://upload.wikimedia.org/wikipedia/en/1/1b/Indiana_Pacers.svg',
    'Los Angeles Clippers': 'https://upload.wikimedia.org/wikipedia/en/thumb/e/ed/Los_Angeles_Clippers_%282024%29.svg/200px-Los_Angeles_Clippers_%282024%29.svg.png',
    'Los Angeles Lakers': 'https://upload.wikimedia.org/wikipedia/commons/3/3c/Los_Angeles_Lakers_logo.svg',
    'Memphis Grizzlies': 'https://upload.wikimedia.org/wikipedia/en/f/f1/Memphis_Grizzlies.svg',
    'Miami Heat': 'https://upload.wikimedia.org/wikipedia/en/f/fb/Miami_Heat_logo.svg',
    'Milwaukee Bucks': 'https://upload.wikimedia.org/wikipedia/en/4/4a/Milwaukee_Bucks_logo.svg',
    'Minnesota Timberwolves': 'https://upload.wikimedia.org/wikipedia/en/c/c2/Minnesota_Timberwolves_logo.svg',
    'New Orleans Pelicans': 'https://upload.wikimedia.org/wikipedia/en/0/0d/New_Orleans_Pelicans_logo.svg',
    'New York Knicks': 'https://upload.wikimedia.org/wikipedia/en/2/25/New_York_Knicks_logo.svg',
    'Oklahoma City Thunder': 'https://upload.wikimedia.org/wikipedia/en/5/5d/Oklahoma_City_Thunder.svg',
    'Orlando Magic': 'https://upload.wikimedia.org/wikipedia/en/thumb/1/10/Orlando_Magic_logo.svg/220px-Orlando_Magic_logo.svg.png',
    'Philadelphia 76ers': 'https://upload.wikimedia.org/wikipedia/en/0/0e/Philadelphia_76ers_logo.svg',
    'Phoenix Suns': 'https://upload.wikimedia.org/wikipedia/en/d/dc/Phoenix_Suns_logo.svg',
    'Portland Trail Blazers': 'https://upload.wikimedia.org/wikipedia/en/2/21/Portland_Trail_Blazers_logo.svg',
    'Sacramento Kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c7/SacramentoKings.svg/180px-SacramentoKings.svg.png',
    'San Antonio Spurs': 'https://upload.wikimedia.org/wikipedia/en/a/a2/San_Antonio_Spurs.svg',
    'Toronto Raptors': 'https://upload.wikimedia.org/wikipedia/en/3/36/Toronto_Raptors_logo.svg',
    'Utah Jazz': 'https://upload.wikimedia.org/wikipedia/en/thumb/5/52/Utah_Jazz_logo_2022.svg/230px-Utah_Jazz_logo_2022.svg.png',
    'Washington Wizards': 'https://upload.wikimedia.org/wikipedia/en/0/02/Washington_Wizards_logo.svg'
}

def get_team_features(team_id, upcoming_date, x=35):
    """
    Fetch recent game logs for a team using nba_api, compute rolling averages for key stats
    over the last x games, and calculate REST_DAYS (days since the last game).
    """
    try:
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2024-25', timeout = 60)
        df = gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching game log for team {team_id}: {e}")
        return None

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format="%b %d, %Y")
    df = df[df['GAME_DATE'] < upcoming_date].sort_values('GAME_DATE')

    if len(df) < x:
        return None

    df_recent = df.tail(x)
    features = {
        'rolling_PTS': df_recent['PTS'].mean(),
        'rolling_FG_PCT': df_recent['FG_PCT'].mean(),
        'rolling_FT_PCT': df_recent['FT_PCT'].mean(),
        'rolling_FG3_PCT': df_recent['FG3_PCT'].mean(),
        'rolling_AST': df_recent['AST'].mean(),
        'rolling_REB': df_recent['REB'].mean()
    }
    last_game_date = df_recent['GAME_DATE'].max()
    features['REST_DAYS'] = (upcoming_date - last_game_date).days

    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability_home = None
    probability_away = None
    team_options = sorted(TEAM_DICT.keys())
    selected_home = None
    selected_away = None
    
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        game_date_str = request.form['game_date']
        
        # Save the selected teams so they persist on render
        selected_home = home_team
        selected_away = away_team
        
        try:
            upcoming_date = pd.to_datetime(game_date_str)
        except Exception:
            upcoming_date = pd.Timestamp.today() + pd.Timedelta(days=1)
        
        home_team_id = TEAM_DICT.get(home_team)
        away_team_id = TEAM_DICT.get(away_team)
        
        if not home_team_id or not away_team_id:
            prediction = "Invalid team selection. Please choose valid teams."
        else:
            home_features = get_team_features(home_team_id, upcoming_date, x=35)
            away_features = get_team_features(away_team_id, upcoming_date, x=35)
            
            if home_features is None or away_features is None:
                prediction = "Not enough data for one of the teams. Try a different team or wait for more games."
            else:
                features = {
                    'home_rolling_PTS': home_features['rolling_PTS'],
                    'home_rolling_FG_PCT': home_features['rolling_FG_PCT'],
                    'home_rolling_FT_PCT': home_features['rolling_FT_PCT'],
                    'home_rolling_FG3_PCT': home_features['rolling_FG3_PCT'],
                    'home_rolling_AST': home_features['rolling_AST'],
                    'home_rolling_REB': home_features['rolling_REB'],
                    'home_REST_DAYS': home_features['REST_DAYS'],
                    'away_rolling_PTS': away_features['rolling_PTS'],
                    'away_rolling_FG_PCT': away_features['rolling_FG_PCT'],
                    'away_rolling_FT_PCT': away_features['rolling_FT_PCT'],
                    'away_rolling_FG3_PCT': away_features['rolling_FG3_PCT'],
                    'away_rolling_AST': away_features['rolling_AST'],
                    'away_rolling_REB': away_features['rolling_REB'],
                    'away_REST_DAYS': away_features['REST_DAYS']
                }
                features['diff_rolling_PTS'] = features['home_rolling_PTS'] - features['away_rolling_PTS']
                features['diff_rolling_FG_PCT'] = features['home_rolling_FG_PCT'] - features['away_rolling_FG_PCT']
                features['diff_rolling_FT_PCT'] = features['home_rolling_FT_PCT'] - features['away_rolling_FT_PCT']
                features['diff_rolling_FG3_PCT'] = features['home_rolling_FG3_PCT'] - features['away_rolling_FG3_PCT']
                features['diff_rolling_AST'] = features['home_rolling_AST'] - features['away_rolling_AST']
                features['diff_rolling_REB'] = features['home_rolling_REB'] - features['away_rolling_REB']
                features['diff_REST_DAYS'] = features['home_REST_DAYS'] - features['away_REST_DAYS']
                
                feature_order = [
                    'home_rolling_PTS', 'home_rolling_FG_PCT', 'home_rolling_FT_PCT', 'home_rolling_FG3_PCT',
                    'home_rolling_AST', 'home_rolling_REB', 'home_REST_DAYS',
                    'away_rolling_PTS', 'away_rolling_FG_PCT', 'away_rolling_FT_PCT', 'away_rolling_FG3_PCT',
                    'away_rolling_AST', 'away_rolling_REB', 'away_REST_DAYS',
                    'diff_rolling_PTS', 'diff_rolling_FG_PCT', 'diff_rolling_FT_PCT', 'diff_rolling_FG3_PCT',
                    'diff_rolling_AST', 'diff_rolling_REB', 'diff_REST_DAYS'
                ]
                expected_features = model.named_steps['scaler'].feature_names_in_
                feature_vector = feature_vector[expected_features]
                
                pred = model.predict(feature_vector)
                proba = model.predict_proba(feature_vector)[0][1]  # probability of home win
                probability_home = round(proba * 100, 2)
                probability_away = round((1 - proba) * 100, 2)
                
                if pred[0] == 1:
                    prediction = "The model predicts that the home team will win."
                else:
                    prediction = "The model predicts that the away team will win."
                    
    return render_template('index.html',
                           prediction=prediction,
                           probability_home=probability_home,
                           probability_away=probability_away,
                           team_options=team_options,
                           team_logos=TEAM_LOGOS,
                           selected_home=selected_home,
                           selected_away=selected_away)

@app.route('/today')
def today():
    # Set the upcoming date to today (using local date in YYYY-MM-DD format)
    upcoming_date = pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d"))
    today_str = datetime.datetime.now().strftime("%m/%d/%Y")


    # Use the ScoreboardV2 endpoint to fetch today's scheduled games
    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today_str)
        games_df = scoreboard.get_data_frames()[0]  # Assuming this DataFrame contains game data
    except Exception as e:
        print("Error fetching today's games:", e)
        games_df = pd.DataFrame()

    # Create an inverse mapping from team IDs to team names for lookup
    TEAM_DICT_INV = {v: k for k, v in TEAM_DICT.items()}

    predictions = []
    if not games_df.empty:
        for i, game in games_df.iterrows():
            # Extract team IDs and convert to string
            home_team_id = str(game['HOME_TEAM_ID'])
            away_team_id = str(game['VISITOR_TEAM_ID'])
            home_team_name = TEAM_DICT_INV.get(home_team_id)
            away_team_name = TEAM_DICT_INV.get(away_team_id)
            if home_team_name and away_team_name:
                home_features = get_team_features(home_team_id, upcoming_date, x=35)
                away_features = get_team_features(away_team_id, upcoming_date, x=35)
                if home_features and away_features:
                    features = {
                        'home_rolling_PTS': home_features['rolling_PTS'],
                        'home_rolling_FG_PCT': home_features['rolling_FG_PCT'],
                        'home_rolling_FT_PCT': home_features['rolling_FT_PCT'],
                        'home_rolling_FG3_PCT': home_features['rolling_FG3_PCT'],
                        'home_rolling_AST': home_features['rolling_AST'],
                        'home_rolling_REB': home_features['rolling_REB'],
                        'home_REST_DAYS': home_features['REST_DAYS'],
                        'away_rolling_PTS': away_features['rolling_PTS'],
                        'away_rolling_FG_PCT': away_features['rolling_FG_PCT'],
                        'away_rolling_FT_PCT': away_features['rolling_FT_PCT'],
                        'away_rolling_FG3_PCT': away_features['rolling_FG3_PCT'],
                        'away_rolling_AST': away_features['rolling_AST'],
                        'away_rolling_REB': away_features['rolling_REB'],
                        'away_REST_DAYS': away_features['REST_DAYS']
                    }
                    features['diff_rolling_PTS'] = features['home_rolling_PTS'] - features['away_rolling_PTS']
                    features['diff_rolling_FG_PCT'] = features['home_rolling_FG_PCT'] - features['away_rolling_FG_PCT']
                    features['diff_rolling_FT_PCT'] = features['home_rolling_FT_PCT'] - features['away_rolling_FT_PCT']
                    features['diff_rolling_FG3_PCT'] = features['home_rolling_FG3_PCT'] - features['away_rolling_FG3_PCT']
                    features['diff_rolling_AST'] = features['home_rolling_AST'] - features['away_rolling_AST']
                    features['diff_rolling_REB'] = features['home_rolling_REB'] - features['away_rolling_REB']
                    features['diff_REST_DAYS'] = features['home_REST_DAYS'] - features['away_REST_DAYS']

                    feature_order = [
                        'home_rolling_PTS', 'home_rolling_FG_PCT', 'home_rolling_FT_PCT', 'home_rolling_FG3_PCT',
                        'home_rolling_AST', 'home_rolling_REB', 'home_REST_DAYS',
                        'away_rolling_PTS', 'away_rolling_FG_PCT', 'away_rolling_FT_PCT', 'away_rolling_FG3_PCT',
                        'away_rolling_AST', 'away_rolling_REB', 'away_REST_DAYS',
                        'diff_rolling_PTS', 'diff_rolling_FG_PCT', 'diff_rolling_FT_PCT', 'diff_rolling_FG3_PCT',
                        'diff_rolling_AST', 'diff_rolling_REB', 'diff_REST_DAYS'
                    ]
                    feature_vector = np.array([features[key] for key in feature_order]).reshape(1, -1)
                    pred = model.predict(feature_vector)
                    proba = model.predict_proba(feature_vector)[0][1]  # probability of home win
                    probability_home = round(proba * 100, 2)
                    probability_away = round((1 - proba) * 100, 2)
                    outcome = "Home Win" if pred[0] == 1 else "Away Win"
                    predictions.append({
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'home_probability': probability_home,
                        'away_probability': probability_away,
                        'prediction': outcome
                    })
                else:
                    predictions.append({
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'error': 'Not enough data for prediction.'
                    })
    else:
        predictions = None

    return render_template(
        'today.html', 
        predictions=predictions, 
        today_date=upcoming_date.strftime("%Y-%m-%d"),
        team_logos=TEAM_LOGOS
    )

if __name__ == '__main__':
    app.run(debug=True)
