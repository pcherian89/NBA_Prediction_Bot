import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
import streamlit as st
from typing import List, Dict
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

# ✅ Streamlit UI Setup
st.set_page_config(page_title="AI-Powered NBA Game Analyzer", layout="wide")
st.title("📊 AI-Powered NBA Game Analyzer")

st.markdown("""
Welcome to the **NBA Analytics Platform**!  
This tool analyzes and predicts NBA game outcomes using AI-powered models trained on historical team and player statistics from the 2024–2025 seasons.
""")

# ✅ Helper Functions
@st.cache_data
def get_teams():
    query = """
        SELECT DISTINCT playerteamname AS team 
        FROM player_stats_filtered 
        WHERE EXTRACT(YEAR FROM gamedate) IN (2024, 2025)
        ORDER BY team
    """
    return pd.read_sql(query, engine)["team"].tolist()

@st.cache_data
def get_players_by_team(team_name):
    if not team_name or not team_name.strip():
        return []
    query = """
        SELECT DISTINCT firstname, lastname 
        FROM player_stats_filtered 
        WHERE playerteamname = %s 
        AND EXTRACT(YEAR FROM gamedate) IN (2024, 2025)
        ORDER BY lastname, firstname
    """
    df = pd.read_sql(query, engine, params=[team_name])
    return df.apply(lambda row: f"{row['firstname']} {row['lastname']}", axis=1).tolist()

def split_name(full_name):
    parts = full_name.strip().split()
    return {"first": parts[0], "last": " ".join(parts[1:])}

# ✅ STEP 1: Select Teams
st.markdown("### 🏀 Select Teams")
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", ["Select a team"] + get_teams())
with col2:
    away_team = st.selectbox("✈️ Away Team", ["Select a team"] + get_teams())

# ✅ STEP 2: Select Players (only after both teams are selected)
home_players_raw = []
away_players_raw = []

if home_team != "Select a team" and away_team != "Select a team":
    st.markdown("### 👤 Select Key Players")

    col1, col2 = st.columns(2)

    try:
        with col1:
            home_players_raw = st.multiselect(
                f"🔵 Select 3 Key Players for {home_team}",
                options=get_players_by_team(home_team),
                max_selections=3
            )
    except Exception as e:
        st.error(f"Error fetching players for {home_team}: {e}")

    try:
        with col2:
            away_players_raw = st.multiselect(
                f"🟠 Select 3 Key Players for {away_team}",
                options=get_players_by_team(away_team),
                max_selections=3
            )
    except Exception as e:
        st.error(f"Error fetching players for {away_team}: {e}")

home_players = [split_name(p) for p in home_players_raw]
away_players = [split_name(p) for p in away_players_raw]

# ✅ Prediction Agent (Assuming GamePredictionAgent class is defined below or imported)

# Only show prediction when everything is ready
if (
    home_team != "Select a team" and away_team != "Select a team" and
    len(home_players) == 3 and len(away_players) == 3
):
    if st.button("🔮 Run Prediction"):
        from nba_agent import GamePredictionAgent  # or include the full class here directly
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
else:
    st.info("👉 Please select both teams and exactly 3 players for each before running predictions.")
