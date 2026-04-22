"""
Streamlit Dashboard for Q-Quant Engine.
Displays single ETF pick from QAOA and VQE, plus top 3 by expected return.
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
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-return { font-size: 2rem; font-weight: 600; }
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

def display_optimizer_tab(optimizer_data: dict, optimizer_name: str):
    top_picks = optimizer_data.get('top_picks', {})
    top3_dict = optimizer_data.get('top3', {})
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            pick = top_picks.get(key)
            if pick:
                ticker = pick['ticker']
                exp_ret = pick['expected_return']
                st.markdown(f"""
                <div class="hero-card">
                    <div style="font-size: 1.2rem; opacity: 0.8;">⚛️ {optimizer_name} TOP PICK</div>
                    <div class="hero-ticker">{ticker}</div>
                    <div class="hero-return">Expected Return: {exp_ret*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Display top 3 table
                top3 = top3_dict.get(key, [])
                if top3:
                    st.markdown("### Top 3 ETFs by Expected Return")
                    df = pd.DataFrame(top3)
                    df['Expected Return'] = df['expected_return'].apply(lambda x: f"{x*100:.2f}%")
                    df = df[['ticker', 'Expected Return']].rename(columns={'ticker': 'Ticker'})
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No selection for {key}.")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">⚛️ P2Quant Q-Quant</div>', unsafe_allow_html=True)
st.markdown('<div>Quantum‑Classical Hybrid – QAOA & VQE for Single ETF Selection</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

main_tab1, main_tab2 = st.tabs(["🌀 QAOA", "⚡ VQE"])

with main_tab1:
    if 'qaoa' in data:
        display_optimizer_tab(data['qaoa'], "QAOA")
    else:
        st.warning("QAOA data not available.")

with main_tab2:
    if 'vqe' in data:
        display_optimizer_tab(data['vqe'], "VQE")
    else:
        st.warning("VQE data not available.")
