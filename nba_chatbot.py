# ✅ FULLY SELF-CONTAINED nba_chatbot.py
# Includes UI + GamePredictionAgent class (restored + with visualizations)

import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
import streamlit as st
from typing import List, Dict
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import pytz
from sqlalchemy import text
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm=ChatOpenAI(temperature=0),


TEAM_NAME_MAPPING = {
    "Golden State Warriors": "Warriors",
    "Los Angeles Lakers": "Lakers",
    "Boston Celtics": "Celtics",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Phoenix Suns": "Suns",
    "New York Knicks": "Knicks",
    "Brooklyn Nets": "Nets",
    "Chicago Bulls": "Bulls",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Philadelphia 76ers": "76ers",
    "Memphis Grizzlies": "Grizzlies",
    "Toronto Raptors": "Raptors",
    "Atlanta Hawks": "Hawks",
    "Cleveland Cavaliers": "Cavaliers",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "New Orleans Pelicans": "Pelicans",
    "San Antonio Spurs": "Spurs",
    "Charlotte Hornets": "Hornets",
    "Orlando Magic": "Magic",
    "Washington Wizards": "Wizards",
    "Sacramento Kings": "Kings",
    "Detroit Pistons": "Pistons",
    "Minnesota Timberwolves": "Timberwolves",
    "Portland Trail Blazers": "Trail Blazers",
    "LA Clippers": "Clippers",
    "Oklahoma City Thunder": "Thunder",
    "Utah Jazz": "Jazz",
}


def fetch_todays_nba_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        response = requests.get(url)
        data = response.json()
        games = []

        for event in data.get("events", []):
            competition = event["competitions"][0]
            home = competition["competitors"][0]
            away = competition["competitors"][1]
            if home["homeAway"] == "away":
                home, away = away, home

            game_time_utc = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            est = pytz.timezone("US/Eastern")
            game_time_est = game_time_utc.astimezone(est).strftime("%I:%M %p")

            games.append({
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "status": competition["status"]["type"]["description"],
                "time": game_time_est
            })
        return games
    except Exception as e:
        st.error(f"Error fetching today's games: {e}")
        return []


# ✅ Theme Colors
HOME_COLOR = "#2ecc71"
AWAY_COLOR = "#e74c3c"

# ✅ Load Models
model_dir = os.path.dirname(__file__)
player_model = joblib.load(os.path.join(model_dir, "player_xgb_model.pkl"))
team_model = joblib.load(os.path.join(model_dir, "team_xgb_model.pkl"))

# ✅ DB Connection
DB_NAME = "postgres"
DB_USER = "nbaadmin"
DB_PASSWORD = "Nbaproject2025"
DB_HOST = "nba-database.c7e4q80iyivf.us-east-1.rds.amazonaws.com"
DB_PORT = "5432"
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ✅ Streamlit Setup
st.set_page_config(page_title="AI-Powered NBA Game Analyzer", layout="wide")
st.title("📊 AI-Powered NBA Game Analyzer")

st.markdown("""
Welcome to the **NBA Analytics Platform**!  
This tool analyzes and predicts NBA game outcomes using AI-powered models trained on historical team and player statistics from the 2024–2025 seasons.
""")

st.markdown("## 🗓️ Today's NBA Games")

games_today = fetch_todays_nba_games()
if games_today:
    for idx, game in enumerate(games_today):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{game['away_team']}** @ **{game['home_team']}** — ⏰ {game['time']} ({game['status']})")
        with col2:
            if st.button("Select", key=f"select_game_{idx}"):
                st.session_state.home_team = TEAM_NAME_MAPPING.get(game["home_team"], "Select a team")
                st.session_state.away_team = TEAM_NAME_MAPPING.get(game["away_team"], "Select a team")


else:
    st.info("No games scheduled for today.")

with st.expander("View NBA Injury Report"):
    try:
        injury_df = pd.read_csv("https://raw.githubusercontent.com/pcherian89/NBA_Prediction_Bot/refs/heads/main/nba-injury-report.csv")
        
        # Optional: Highlight players who are OUT
        def highlight_out(s):
            return ['background-color: #ffcccc' if v.lower() == 'out' else '' for v in s]

        styled_df = injury_df.style.apply(highlight_out, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Could not load injury report: {e}")



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
        SELECT firstname, lastname 
        FROM player_stats_filtered 
        WHERE playerteamname = %s 
        AND EXTRACT(YEAR FROM gamedate) IN (2024, 2025)
    """
    df = pd.read_sql(query, engine, params=(team_name,))
    df = df.dropna(subset=["firstname", "lastname"])
    df["fullname"] = df["firstname"].astype(str).str.strip() + " " + df["lastname"].astype(str).str.strip()
    player_list = df["fullname"].drop_duplicates().dropna().sort_values().tolist()
    return [p for p in player_list if isinstance(p, str) and len(p.strip()) > 0]

def split_name(full_name):
    parts = full_name.strip().split()
    return {"first": parts[0], "last": " ".join(parts[1:])}

# ✅ Select Teams
st.markdown("### 🏀 Select Teams")
teams_list = ["Select a team"] + get_teams()

home_team = st.selectbox(
    "🏠 Home Team",
    teams_list,
    index=teams_list.index(st.session_state.get("home_team", "Select a team")),
    key="home_team"
)

away_team = st.selectbox(
    "✈️ Away Team",
    teams_list,
    index=teams_list.index(st.session_state.get("away_team", "Select a team")),
    key="away_team"
)

home_players_raw = []
away_players_raw = []

if home_team != "Select a team" and away_team != "Select a team":
    st.markdown("### 👤 Select Key Players")
    col1, col2 = st.columns(2)
    try:
        with col1:
            home_players_raw = st.multiselect(
                f"🔵 Select Key Players for {home_team} (Pick 3)",
                options=get_players_by_team(home_team)
            )
    except Exception as e:
        st.error(f"Error fetching players for {home_team}: {e}")

    try:
        with col2:
            away_players_raw = st.multiselect(
                f"🟠 Select Key Players for {away_team} (Pick 3)",
                options=get_players_by_team(away_team)
            )
    except Exception as e:
        st.error(f"Error fetching players for {away_team}: {e}")

home_players = [split_name(p) for p in home_players_raw]
away_players = [split_name(p) for p in away_players_raw]

# ✅ GamePredictionAgent class (with full visuals)
class GamePredictionAgent:
    def __init__(self):
        self.last_home_player_stats = None
        self.last_away_player_stats = None
        self.last_home_team_df = None
        self.last_away_team_df = None

    

    def get_player_data(self, players: List[Dict[str, str]], team_name: str) -> pd.DataFrame:
        if not players:
            return pd.DataFrame()

        # Build WHERE clause
        conditions = []
        for idx, p in enumerate(players):
            conditions.append(f"(firstname ILIKE :first{idx} AND lastname ILIKE :last{idx})")

        where_clause = " OR ".join(conditions)
        full_query = f"""
            SELECT * FROM player_stats_filtered
            WHERE playerteamname ILIKE :team AND ({where_clause})
            ORDER BY gamedate DESC
        """

        # Prepare parameter dictionary
        param_dict = {"team": team_name}
        for idx, p in enumerate(players):
            param_dict[f"first{idx}"] = p["first"]
            param_dict[f"last{idx}"] = p["last"]

        with engine.connect() as conn:
            result = conn.execute(text(full_query), param_dict)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        return df




    def predict_player_win_probability(self, players, team_name, store_attr):
        df = self.get_player_data(players, team_name)
        if df.empty:
            setattr(self, store_attr, pd.DataFrame())
            return {"team_average_win_probability": 0.0}

        df = df.sort_values(by=["firstname", "lastname", "gamedate"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["firstname", "lastname", "gameid"])
        df = df.groupby(["firstname", "lastname"]).head(20)

        model_features = ['home', 'numminutes', 'points', 'assists', 'blocks', 'steals',
                          'fieldgoalspercentage', 'reboundstotal', 'turnovers', 'plusminuspoints',
                          'true_shooting_percentage', 'usage_rate', 'assist_percentage',
                          'turnover_percentage', 'rebound_percentage']

        df_features = df[model_features].copy()
        df_features = df_features.fillna(0)
        df["win_probability"] = player_model.predict_proba(df_features)[:, 1]

        player_stats = df.groupby(["firstname", "lastname"]).agg({
            "win_probability": "mean",
            **{col: "mean" for col in model_features}
        }).reset_index()

        setattr(self, store_attr, player_stats)

        st.markdown(f"#### 🔍 Win Probabilities for {team_name} Players")
        st.dataframe(player_stats[["firstname", "lastname", "win_probability"] + model_features])
        return {"team_average_win_probability": round(player_stats["win_probability"].mean(), 4)}

    def get_team_data(self, team_name):
        query = f"""
        SELECT * FROM team_stats_filtered
        WHERE teamid = (
            SELECT DISTINCT teamid FROM team_stats_filtered
            WHERE teamname ILIKE '{team_name}' LIMIT 1
        )
        ORDER BY gamedate DESC;
        """
        return pd.read_sql(query, engine)

    def predict_team_win_probability(self, team_name, store_attr):
        df = self.get_team_data(team_name)
        if df.empty:
            setattr(self, store_attr, pd.DataFrame())
            return {"average_team_win_probability": 0.0}

        df = df.drop_duplicates(subset=["gameid"]).head(10)
        feature_cols = ['ast_to_ratio_home', 'reb_percentage_home', 'efg_percentage_home', 'free_throw_rate']
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(df[feature_cols].median())
        df["win_probability"] = team_model.predict_proba(df[feature_cols])[:, 1]

        setattr(self, store_attr, df)

        st.markdown(f"#### 🔍 {team_name} - Last 10 Games")
        st.dataframe(df[["gamedate", "win", "win_probability"] + feature_cols])

        return {"average_team_win_probability": round(df["win_probability"].mean(), 4)}

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
            ax.bar(df["player"], df[metric], color=[HOME_COLOR if t == "Home" else AWAY_COLOR for t in df["team"]])
            ax.set_title(f"{metric} (Avg over last 20 games)", fontsize=12)
            ax.set_ylabel(metric)
            ax.set_ylim(0, df[metric].max() * 1.2)
            ax.grid(True)
            st.pyplot(fig)

    def plot_team_metric_trends(self, home_team, away_team):
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
            fig.autofmt_xdate()
            st.pyplot(fig)

    def predict_game(self, home_team, away_team, home_players, away_players):
        home_player_prob = self.predict_player_win_probability(home_players, home_team, store_attr="last_home_player_stats")
        away_player_prob = self.predict_player_win_probability(away_players, away_team, store_attr="last_away_player_stats")
        home_team_prob = self.predict_team_win_probability(home_team, store_attr="last_home_team_df")
        away_team_prob = self.predict_team_win_probability(away_team, store_attr="last_away_team_df")

        home = round(0.5 * home_player_prob["team_average_win_probability"] + 0.5 * home_team_prob["average_team_win_probability"], 4)
        away = round(0.5 * away_player_prob["team_average_win_probability"] + 0.5 * away_team_prob["average_team_win_probability"], 4)
        total = home + away
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_final_probability": round(home / total, 2),
            "away_final_probability": round(away / total, 2)
        }

# ✅ Run Prediction
if (
    home_team != "Select a team" and away_team != "Select a team" and
    len(home_players) == 3 and len(away_players) == 3
):
    if st.button("🔮 Run Prediction"):
        agent = GamePredictionAgent()
        result = agent.predict_game(home_team, away_team, home_players, away_players)

        # 🧠 Store for chatbot context
        st.session_state.agent = agent
        st.session_state.prediction_result = result

        st.success("✅ Prediction complete!")
        st.markdown("### 🎯 Final Win Probabilities")
        st.markdown(f"<h3 style='text-align: center;'>{result['home_team']} (Home): <span style='color:{HOME_COLOR}'>{result['home_final_probability']:.2f}</span> &nbsp;&nbsp;|&nbsp;&nbsp; {result['away_team']} (Away): <span style='color:{AWAY_COLOR}'>{result['away_final_probability']:.2f}</span></h3>", unsafe_allow_html=True)

# ✅ Display visuals and insights if prediction exists
if "prediction_result" in st.session_state and "agent" in st.session_state:
    agent = st.session_state.agent
    result = st.session_state.prediction_result

    st.markdown("### 📊 Win Probabilities for Warriors Players")
    st.dataframe(agent.last_home_player_stats)

    st.markdown("### 📊 Win Probabilities for Lakers Players")
    st.dataframe(agent.last_away_player_stats)

    st.markdown("### 🧠 Warriors - Last 10 Games")
    st.dataframe(agent.last_home_team_df)

    st.markdown("### 🧠 Lakers - Last 10 Games")
    st.dataframe(agent.last_away_team_df)

    # 🎲 Convert Win % to Odds
    def win_prob_to_decimal_odds(prob):
        return round(1 / prob, 2)

    def win_prob_to_american_odds(prob):
        if prob >= 0.5:
            return f"-{round(prob / (1 - prob) * 100):.0f}"
        else:
            return f"+{round((1 - prob) / prob * 100):.0f}"

    home_decimal = win_prob_to_decimal_odds(result["home_final_probability"])
    away_decimal = win_prob_to_decimal_odds(result["away_final_probability"])
    home_american = win_prob_to_american_odds(result["home_final_probability"])
    away_american = win_prob_to_american_odds(result["away_final_probability"])

    # 💰 Odds Breakdown
    st.markdown("### 💰 Odds Breakdown")
    odds_cols = st.columns([3, 1.5, 2, 2])
    header_style = "font-weight: 600; font-size: 15px"
    odds_cols[0].markdown(f"<div style='{header_style}'>Team</div>", unsafe_allow_html=True)
    odds_cols[1].markdown(f"<div style='{header_style}'>Win %</div>", unsafe_allow_html=True)
    odds_cols[2].markdown(f"<div style='{header_style}'>Decimal Odds</div>", unsafe_allow_html=True)
    odds_cols[3].markdown(f"<div style='{header_style}'>Odds</div>", unsafe_allow_html=True)

    # Row 1 – Home
    odds_cols = st.columns([3, 1.5, 2, 2])
    odds_cols[0].markdown(f"**{result['home_team']} (Home)**")
    odds_cols[1].markdown(f"{result['home_final_probability']:.0%}")
    odds_cols[2].markdown(f"{home_decimal:.2f}")
    odds_cols[3].markdown(f"{home_american}")

    # Row 2 – Away
    odds_cols = st.columns([3, 1.5, 2, 2])
    odds_cols[0].markdown(f"**{result['away_team']} (Away)**")
    odds_cols[1].markdown(f"{result['away_final_probability']:.0%}")
    odds_cols[2].markdown(f"{away_decimal:.2f}")
    odds_cols[3].markdown(f"{away_american}")

    # 📈 Charts and Trends
    with st.expander("📈 Team Metric Trends Over Last 10 Games"):
        agent.plot_team_metric_trends(home_team, away_team)

    with st.expander("📊 View Player Feature Comparison Charts"):
        agent.plot_player_win_probs()
        agent.plot_player_feature_comparisons()

    st.markdown("---")
    st.markdown("📊 Run again above to test another matchup!")

    # ✅ Insert LangChain Chatbot (AFTER all visuals)
    st.markdown("---")
    with st.expander("💬 Ask the NBA Bot About This Matchup", expanded=True):
    
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_input" not in st.session_state:
            st.session_state.chat_input = ""
    
        st.session_state.chat_input = st.text_input(
            "🧠 Type your question:",
            value=st.session_state.chat_input,
            key="chat_input_field"
        )
    
        if st.session_state.chat_input and "agent" in st.session_state:
    
            from langchain_experimental.agents import create_pandas_dataframe_agent
            from langchain_openai import OpenAI
    
            agent = st.session_state.agent
    
            df_home = agent.last_home_player_stats.copy()
            df_away = agent.last_away_player_stats.copy()
            df_home["player_name"] = df_home["firstname"] + " " + df_home["lastname"]
            df_away["player_name"] = df_away["firstname"] + " " + df_away["lastname"]
            df_home["source"] = "home_players"
            df_away["source"] = "away_players"
    
            df_team_home = agent.last_home_team_df.copy()
            df_team_away = agent.last_away_team_df.copy()
            df_team_home["source"] = "home_team"
            df_team_away["source"] = "away_team"
    
            combined_df = pd.concat([df_home, df_away, df_team_home, df_team_away], ignore_index=True)
    
            chatbot = create_pandas_dataframe_agent(
                llm=ChatOpenAI(temperature=0),
                df=combined_df,
                verbose=False,
                allow_dangerous_code=True
            )
                
            response = chatbot.run(st.session_state.chat_input)
            st.session_state.chat_history.append(("You", st.session_state.chat_input))
            st.session_state.chat_history.append(("Bot", response))
            st.session_state.chat_input = ""
    
        # Display chat history
        for role, msg in st.session_state.chat_history[::-1]:
            bg = "#f1f1f1" if role == "You" else "#d1f5d3"
            icon = "🧍" if role == "You" else "🤖"
            st.markdown(f"""
            <div style='background-color:{bg};padding:10px;border-radius:10px;margin-bottom:5px'>
            <b>{icon} {role}:</b> {msg}
            </div>
            """, unsafe_allow_html=True)

# 👇 Input validation fallback
else:
    st.info("👉 Please select both teams and exactly 3 players for each before running predictions.")


