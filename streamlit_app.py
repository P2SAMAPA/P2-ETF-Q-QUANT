"""
Streamlit Dashboard for Q-Quant Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Q-Quant", page_icon="⚛️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; }
    .ticker-list { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">⚛️ P2Quant Q-Quant</div>', unsafe_allow_html=True)
st.markdown('<div>Quantum‑Classical Hybrid – QAOA for ETF Portfolio Selection</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        universe_data = daily['universes'].get(key, {})
        selected = universe_data.get('selected_tickers', [])
        port_return = universe_data.get('portfolio_return', 0.0)
        port_risk = universe_data.get('portfolio_risk', 0.0)

        if selected:
            st.markdown(f"""
            <div class="hero-card">
                <h2>⚛️ QAOA-Selected Portfolio</h2>
                <div class="ticker-list">Selected ETFs: {', '.join(selected)}</div>
                <p>Expected Return: {port_return*100:.2f}%</p>
                <p>Expected Risk: {port_risk*100:.2f}%</p>
                <p>Sharpe Ratio: {port_return/port_risk:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Selected ETFs")
            df = pd.DataFrame({'Ticker': selected})
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No portfolio selected for this universe.")
