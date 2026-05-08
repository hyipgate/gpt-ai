from __future__ import annotations

import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import load_config, resolve_path


st.set_page_config(page_title="AI SMC Trader", layout="wide")
st.title("AI SMC Trader")

config = load_config()
db_path = resolve_path(config.paths.database_url.replace("sqlite:///", ""))


@st.cache_data(ttl=10)
def read_table(name: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)


trades = read_table("trades")
equity = read_table("equity_snapshots")

col1, col2, col3, col4 = st.columns(4)
if trades.empty:
    col1.metric("Trades", 0)
    col2.metric("Winrate", "0.0%")
    col3.metric("Open", 0)
    col4.metric("Avg Confidence", "0.00")
else:
    col1.metric("Trades", len(trades))
    closed = trades[trades["status"].isin(["paper_closed", "closed", "filled_closed"]) | trades["profit"].notna()]
    winrate = (closed["profit"] > 0).mean() if not closed.empty else 0
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Open", int(trades["status"].isin(["paper_open", "submitted", "filled"]).sum()))
    col4.metric("Paper PnL", f"${closed['profit'].fillna(0).sum():.2f}" if not closed.empty else "$0.00")

left, right = st.columns([2, 1])
with left:
    st.subheader("Equity Curve")
    if not equity.empty:
        st.plotly_chart(px.line(equity, x="created_at", y="equity"), use_container_width=True)
    elif not trades.empty and "equity" in trades:
        st.plotly_chart(px.line(trades.dropna(subset=["equity"]), x="created_at", y="equity"), use_container_width=True)
    else:
        st.info("No equity snapshots yet.")

with right:
    st.subheader("Drawdown")
    if not equity.empty and "drawdown" in equity:
        st.plotly_chart(px.area(equity, x="created_at", y="drawdown"), use_container_width=True)
    else:
        st.info("No drawdown records yet.")

st.subheader("Confidence Heatmap")
if not trades.empty:
    heat = trades.copy()
    heat["created_at"] = pd.to_datetime(heat["created_at"])
    heat["hour"] = heat["created_at"].dt.hour
    heat["date"] = heat["created_at"].dt.date
    pivot = heat.pivot_table(index="date", columns="hour", values="confidence", aggfunc="mean")
    st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis"), use_container_width=True)
else:
    st.info("No model decisions logged yet.")

st.subheader("Trade Journal")
st.dataframe(trades, use_container_width=True, hide_index=True)
