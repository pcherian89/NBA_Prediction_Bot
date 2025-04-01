# nba_chatbot.py

import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
import streamlit as st
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# ✅ Theme Colors
HOME_COLOR = "#2ecc71"  # green
AWAY_COLOR = "#e74c3c"  # red

# ✅ Load Models
model_dir = os.path.dirname(__file__)
player_model = joblib.load(os.path.join(model_dir, "player_xgb_model.pkl"))
team_model = joblib.load(os.path.join(model_dir, "team_xgb_model.pkl"))

# ✅ Database Connection
DB_NAME = "postgres"
DB_USER = "nbaadmin"
DB_PASSWORD = "Nbaproject2025"
DB_HOST = "nba-database.c7e4q80iyivf.us-east-1.rds.amazonaws.com"
DB_PORT = "5432"
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ✅ Game Prediction Agent
class GamePredictionAgent:
    def __init__(self):
        self.last_home_player_stats = None
        self.last_away_player_stats = None
        self.last_home_team_df = None
        self.last_away_team_df = None

    def get_player_data(self, players: List[Dict[str, str]], team_name: str) -> pd.DataFrame:
        conditions = " OR ".join([f"(firstname ILIKE '{p['first']}' AND lastname ILIKE '{p['last']}')" for p in players])
        query = f"""
        SELECT gameid, firstname, lastname, playerteamname, numminutes, home, win, points, assists, blocks, steals,
               fieldgoalspercentage, reboundstotal, turnovers, plusminuspoints, 
               true_shooting_percentage, usage_rate, assist_percentage, 
               turnover_percentage, rebound_percentage, gamedate
        FROM player_stats_filtered
        WHERE playerteamname ILIKE '{team_name}' AND ({conditions})
        ORDER BY gamedate DESC;
        """
        return pd.read_sql(query, engine)

    def predict_player_win_probability(self, players: List[Dict[str, str]], team_name: str, store_attr: str) -> Dict[str, float]:
        df = self.get_player_data(players, team_name)
        if df.empty:
            setattr(self, store_attr, pd.DataFrame())
            return {"team_average_win_probability": 0.0}

        df["gamedate"] = pd.to_datetime(df["gamedate"])
        df = df.sort_values(by=["firstname", "lastname", "gamedate"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["firstname", "lastname", "gameid"])
        df = df.groupby(["firstname", "lastname"]).head(20)

        win_record = df.groupby(["firstname", "lastname"])["win"].agg(
            games_played="count", wins="sum"
        ).reset_index()
        win_record["losses"] = win_record["games_played"] - win_record["wins"]

        model_features = ['home', 'numminutes', 'points', 'assists', 'blocks', 'steals',
                          'fieldgoalspercentage', 'reboundstotal', 'turnovers', 'plusminuspoints',
                          'true_shooting_percentage', 'usage_rate', 'assist_percentage',
                          'turnover_percentage', 'rebound_percentage']

        df_features = df[model_features].copy()
        df_features[[col for col in model_features if col != 'fieldgoalspercentage']] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(df_features[[col for col in model_features if col != 'fieldgoalspercentage']])
        df_features[['fieldgoalspercentage']] = SimpleImputer(strategy="median").fit_transform(df_features[['fieldgoalspercentage']])

        df["win_probability"] = player_model.predict_proba(df_features)[:, 1]

        player_stats = df.groupby(["firstname", "lastname"]).agg({
            "win_probability": "mean",
            "numminutes": "count",
            **{col: "mean" for col in model_features}
        }).reset_index()
        player_stats = player_stats.merge(win_record, on=["firstname", "lastname"])

        setattr(self, store_attr, player_stats)

        st.markdown(f"#### 🔍 Win Probabilities + Game Records for Key Players ({team_name}):")
        for _, row in player_stats.iterrows():
            st.write(f"🏀 {row['firstname']} {row['lastname']} - Win Prob: {row['win_probability']:.4f} | Games: {row['games_played']} | Wins: {row['wins']} | Losses: {row['losses']}")

        st.markdown(f"#### 📊 Player Feature Averages (Last 20 Games):")
        st.dataframe(player_stats[["firstname", "lastname", "games_played", "wins", "losses", "win_probability"] + model_features])

        return {"team_average_win_probability": round(player_stats["win_probability"].mean(), 4)}

    def get_team_data(self, team_name: str) -> pd.DataFrame:
        query = f"""
        SELECT gameid, teamid, teamname, win, gamedate,
               ast_to_ratio_home, reb_percentage_home, efg_percentage_home, free_throw_rate
        FROM team_stats_filtered
        WHERE teamid = (
            SELECT DISTINCT teamid FROM team_stats_filtered 
            WHERE teamname ILIKE '{team_name}' LIMIT 1
        )
        ORDER BY gamedate DESC;
        """
        return pd.read_sql(query, engine)

    def predict_team_win_probability(self, team_name: str, store_attr: str) -> Dict[str, float]:
        df = self.get_team_data(team_name)
        if df.empty:
            setattr(self, store_attr, pd.DataFrame())
            return {"average_team_win_probability": 0.0}

        df = df.drop_duplicates(subset=["gameid"]).head(10)
        feature_cols = ['ast_to_ratio_home', 'reb_percentage_home', 'efg_percentage_home', 'free_throw_rate']
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(df[feature_cols].median())
        df["win_probability"] = team_model.predict_proba(df[feature_cols])[:, 1]

        setattr(self, store_attr, df)

        st.markdown(f"#### 🔍 {team_name} - Game-by-Game Win Probabilities and Metrics (Last 10 Games):")
        st.dataframe(df[["gamedate", "gameid", "win", "win_probability"] + feature_cols])

        avg_prob = round(df["win_probability"].mean(), 4)
        st.markdown(f"#### 📊 {team_name} Summary: {int(df['win'].sum())} Wins | {int(len(df) - df['win'].sum())} Losses | Avg Win Probability: {avg_prob}")

        return {"average_team_win_probability": avg_prob}

    def plot_player_win_probs(self):
        df = pd.concat([
            self.last_home_player_stats.assign(team="Home"),
            self.last_away_player_stats.assign(team="Away")
        ])
        df["player"] = df["firstname"] + " " + df["lastname"]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df["player"], df["win_probability"], color=[HOME_COLOR if t == "Home" else AWAY_COLOR for t in df["team"]])
        ax.set_ylim(0, 1)
        ax.set_title("Win Probabilities by Player")
        ax.grid(True)
        st.pyplot(fig)

    def plot_player_feature_comparisons(self):
        df = pd.concat([
            self.last_home_player_stats.assign(team="Home"),
            self.last_away_player_stats.assign(team="Away")
        ])
        df["player"] = df["firstname"] + " " + df["lastname"]
        metrics = [
            "plusminuspoints", "true_shooting_percentage", "usage_rate",
            "assist_percentage", "turnover_percentage", "rebound_percentage"
        ]
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(
                df["player"],
                df[metric],
                color=[HOME_COLOR if t == "Home" else AWAY_COLOR for t in df["team"]]
            )
            ax.set_title(f"{metric} (Avg over last 20 games)", fontsize=12)
            ax.set_ylabel(metric)
            ax.set_ylim(0, df[metric].max() * 1.2)
            ax.grid(True)
            st.pyplot(fig)

    def plot_team_metric_trends(self, home_team: str, away_team: str):
        df1 = self.last_home_team_df.copy()
        df2 = self.last_away_team_df.copy()
        df1["gamedate"] = pd.to_datetime(df1["gamedate"])
        df2["gamedate"] = pd.to_datetime(df2["gamedate"])
        df1 = df1.sort_values("gamedate")
        df2 = df2.sort_values("gamedate")
        metrics = ["efg_percentage_home", "reb_percentage_home", "ast_to_ratio_home", "free_throw_rate"]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df1["gamedate"], df1[metric], label=home_team, marker="o", color=HOME_COLOR)
            ax.plot(df2["gamedate"], df2[metric], label=away_team, marker="s", color=AWAY_COLOR)
            ax.set_title(f"{metric} Over Last 10 Games")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    def predict_game(self, home_team: str, away_team: str, home_players: List[Dict[str, str]], away_players: List[Dict[str, str]]) -> Dict:
        home_player_prob = self.predict_player_win_probability(home_players, home_team, store_attr="last_home_player_stats")
        away_player_prob = self.predict_player_win_probability(away_players, away_team, store_attr="last_away_player_stats")
        home_team_prob = self.predict_team_win_probability(home_team, store_attr="last_home_team_df")
        away_team_prob = self.predict_team_win_probability(away_team, store_attr="last_away_team_df")

        home = round(0.5 * home_player_prob["team_average_win_probability"] + 0.5 * self.last_home_team_df["win_probability"].mean(), 4)
        away = round(0.5 * away_player_prob["team_average_win_probability"] + 0.5 * self.last_away_team_df["win_probability"].mean(), 4)
        total = home + away
        home_norm = round(home / total, 2)
        away_norm = round(away / total, 2)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_final_probability": home_norm,
            "away_final_probability": away_norm
        }

# ✅ Streamlit UI
st.set_page_config(page_title="NBA Prediction Bot", layout="wide")
st.title("🏀 NBA Prediction BOT")

st.markdown("""
Welcome to the **NBA Analytics Platform**!  
This tool predicts the outcome of NBA games using AI-powered models trained on team and player statistics.
""")

home_team = st.text_input("🏠 Enter the home team:")
away_team = st.text_input("✈️ Enter the away team:")

st.markdown("### 🔵 Enter 3 Key Players for Home Team")
home_players = []
for i in range(3):
    col1, col2 = st.columns(2)
    with col1:
        first = st.text_input(f"Home Player {i+1} First Name", key=f"home_first_{i}")
    with col2:
        last = st.text_input(f"Home Player {i+1} Last Name", key=f"home_last_{i}")
    home_players.append({"first": first, "last": last})

st.markdown("### 🟠 Enter 3 Key Players for Away Team")
away_players = []
for i in range(3):
    col1, col2 = st.columns(2)
    with col1:
        first = st.text_input(f"Away Player {i+1} First Name", key=f"away_first_{i}")
    with col2:
        last = st.text_input(f"Away Player {i+1} Last Name", key=f"away_last_{i}")
    away_players.append({"first": first, "last": last})

if st.button("🔮 Run Prediction"):
    agent = GamePredictionAgent()
    result = agent.predict_game(home_team, away_team, home_players, away_players)

    st.success("✅ Prediction complete!")

    st.markdown("### 🎯 Final Win Probabilities")
    st.markdown(f"<h3 style='text-align: center;'>{result['home_team']} (Home): <span style='color:{HOME_COLOR}'>{result['home_final_probability']:.2f}</span> &nbsp;&nbsp;|&nbsp;&nbsp; {result['away_team']} (Away): <span style='color:{AWAY_COLOR}'>{result['away_final_probability']:.2f}</span></h3>", unsafe_allow_html=True)

    with st.expander("📈 Team Metric Trends Over Last 10 Games"):
        agent.plot_team_metric_trends(home_team, away_team)

    with st.expander("📊 View Player Feature Comparison Charts"):
        agent.plot_player_win_probs()
        agent.plot_player_feature_comparisons()

    st.markdown("---")
    st.markdown("📊 Run again above to test another matchup!")

