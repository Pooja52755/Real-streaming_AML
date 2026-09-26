import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import textwrap
import importlib
import fraud_data
import graph_vis



# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────
st.html(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp { background-color: #f1f5f9; }

    /* ── Header ── */
    .header-title { font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }
    .header-subtitle { font-size: 13px; color: #64748b; margin-top: 2px; }
    .header-timestamp { font-size: 13px; color: #64748b; }

    /* ── Badges ── */
    .badge-high {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-medium {
        background:#fffbeb; color:#d97706; border:1px solid #fde68a;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-low {
        background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }

    /* ── Textarea styling ── */
    .stTextArea textarea {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-size: 13px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* ── Left Panel: Flagged Account Cards ── */
    .flagged-card {
        background:#ffffff; border:1.5px solid #e2e8f0; border-radius:10px;
        padding:12px 14px; margin-bottom:10px; cursor:pointer;
        transition: box-shadow 0.2s;
        border-left: 4px solid #e2e8f0;
    }
    .flagged-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .flagged-card-high { border-left-color: #ef4444 !important; }
    .flagged-card-medium { border-left-color: #f59e0b !important; }
    .flagged-card-low { border-left-color: #16a34a !important; }
    .flagged-card-selected { background:#eff6ff; border-color:#93c5fd; }
    .flagged-acc-id { font-weight:700; font-size:13px; color:#1e40af; }
    .flagged-pattern { font-size:11px; color:#64748b; margin-top:3px; }
    .risk-score-bar-bg {
        background:#f1f5f9; border-radius:4px; height:6px; margin-top:8px; overflow:hidden;
    }
    .risk-score-bar-fill {
        height:6px; border-radius:4px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }

    /* ── Center Panel ── */
    .center-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── XAI Explanation box ── */
    .xai-box {
        background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
        padding:14px 16px; margin-top:16px;
    }
    .xai-box-high {
        background:#fff5f5; border:1px solid #fecaca;
    }
    .xai-title { font-weight:700; font-size:13px; color:#1e293b; margin-bottom:8px; }
    .xai-list { margin:0; padding-left:18px; font-size:12px; color:#334155; line-height:1.8; }

    /* ── AI Advisory banner ── */
    .ai-advisory {
        background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
        padding:10px 14px; font-size:11.5px; color:#0369a1;
        margin-top:12px; display:flex; align-items:flex-start; gap:8px;
    }

    /* ── Human Decision Panel ── */
    .human-decision-panel {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        border:1.5px solid #fbbf24; border-radius:12px;
        padding:18px 20px; margin-top:18px;
    }
    .human-decision-title {
        font-weight:800; font-size:14px; color:#92400e;
        display:flex; align-items:center; gap:8px; margin-bottom:6px;
    }
    .human-decision-disclaimer {
        font-size:11px; color:#78716c; background:#ffffff;
        border:1px solid #e7e5e4; border-radius:6px;
        padding:8px 12px; margin-top:10px;
    }

    /* ── Right Panel: Customer Profile ── */
    .profile-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
        height:100%;
    }
    .profile-empty {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; padding:40px 20px; text-align:center;
        color:#94a3b8;
    }
    .profile-metric-grid {
        display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px;
    }
    .profile-metric-card {
        background:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:10px 12px;
    }
    .profile-metric-label { font-size:10px; color:#64748b; text-transform:uppercase; font-weight:600; }
    .profile-metric-val { font-size:15px; font-weight:700; color:#0f172a; margin-top:2px; }

    /* ── Financial Summary table ── */
    .custom-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
    .custom-table th {
        text-align:left; padding:8px 10px; background:#f8fafc;
        color:#64748b; font-weight:600; border-bottom:1px solid #e2e8f0;
    }
    .custom-table td { padding:8px 10px; border-bottom:1px solid #f1f5f9; color:#1e293b; }

    /* ── Top banner ── */
    .top-banner {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:16px 22px; display:flex; align-items:center; gap:20px;
        margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .top-banner-icon {
        width:46px; height:46px; background:#2563eb; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        color:white; font-size:22px;
    }
    .banner-value { font-size:26px; font-weight:800; color:#0f172a; line-height:1.2; }
    .banner-subtext { font-size:12px; color:#16a34a; font-weight:600; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e2e8f0; }
    .sidebar-brand {
        display:flex; align-items:center; gap:10px;
        padding:10px 0 20px 0; border-bottom:1px solid #f1f5f9; margin-bottom:20px;
    }
    .sidebar-brand-title { font-size:16px; font-weight:800; color:#0f172a; }
    .sidebar-brand-subtitle { font-size:11px; color:#64748b; }

    /* ── Section labels ── */
    .section-label {
        font-size:11px; font-weight:700; color:#64748b;
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;
    }
</style>
"""))

# ─── Session State Initialization ────────────────────────────────────────
if "selected_tx_id" not in st.session_state:
    st.session_state.selected_tx_id = None
if "selected_sub_tx" not in st.session_state:
    st.session_state.selected_sub_tx = None
if "human_decision_submitted" not in st.session_state:
    st.session_state.human_decision_submitted = {}
if "goto_graph" not in st.session_state:
    st.session_state.goto_graph = False

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(textwrap.dedent("""
    <div class="sidebar-brand">
        <div style="width:34px;height:34px;background:#2563eb;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">
            🛡️
        </div>
        <div>
            <div class="sidebar-brand-title">AML Fraud Detection</div>
            <div class="sidebar-brand-subtitle">GAT-based Transaction Monitoring</div>
        </div>
    </div>
    """))

    default_page_idx = 1 if st.session_state.goto_graph else 0
    page = st.radio(
        "Navigation",
        ["Dashboard", "Graph Network"],
        index=default_page_idx,
        label_visibility="collapsed"
    )
    if st.session_state.goto_graph and page == "Graph Network":
        st.session_state.goto_graph = False

    # ── Live Streaming Simulation Controls ──
    st.markdown("---")
    st.markdown("<div style='font-size:12px;font-weight:700;color:#0f172a;margin-bottom:4px;'>⚡ Real-Time Stream Engine</div>", unsafe_allow_html=True)
    status = fraud_data.get_stream_status()
    curr_tx = status["current_idx"]
    tot_tx = status["total_txs"]
    active_cnt = status["active_investigations"]

    st.markdown(
        f"<div style='font-size:11px;color:#64748b;margin-bottom:6px;'>"
        f"Streamed: <b>{curr_tx} / {tot_tx}</b> txs<br>"
        f"Active Alerts: <b style='color:#dc2626;'>{active_cnt}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.progress(min(1.0, curr_tx / max(1, tot_tx)))

    btn_s1, btn_s2 = st.columns(2)
    with btn_s1:
        if st.button("▶ Step +1", key="btn_stream_1", use_container_width=True, disabled=(curr_tx >= tot_tx)):
            fraud_data.step_stream(1)
            st.rerun()
    with btn_s2:
        if st.button("⏩ Step +10", key="btn_stream_10", use_container_width=True, disabled=(curr_tx >= tot_tx)):
            fraud_data.step_stream(10)
            st.rerun()

    col_live1, col_live2 = st.columns(2)
    with col_live1:
        if st.button("⏱️ Stream 5 Live", key="btn_stream_5_live", use_container_width=True, disabled=(curr_tx >= tot_tx), help="Streams 5 transactions one-by-one with 1 second delay"):
            steps = min(5, tot_tx - curr_tx)
            with st.status(f"⚡ Streaming {steps} txs live (1s/tx)...", expanded=True) as s:
                for i in range(steps):
                    tx = fraud_data.step_stream(1)
                    stat = fraud_data.get_stream_status()
                    last = stat.get("last_tx") or {}
                    fo_str = "🚨 Fan-out!" if last.get("is_fanout") else "1-hop"
                    s.write(f"Tx {stat['current_idx']}/{tot_tx}: `{last.get('from_account')}` ➔ `{last.get('to_account')}` ({last.get('amount_formatted')}) | {fo_str}")
                    time.sleep(1.0)
                s.update(label=f"✅ {steps} Transactions Ingested!", state="complete", expanded=False)
            st.rerun()
    with col_live2:
        if st.button("⏱️ Stream 10 Live", key="btn_stream_10_live", use_container_width=True, disabled=(curr_tx >= tot_tx), help="Streams 10 transactions one-by-one with 1 second delay"):
            steps = min(10, tot_tx - curr_tx)
            with st.status(f"⚡ Streaming {steps} txs live (1s/tx)...", expanded=True) as s:
                for i in range(steps):
                    tx = fraud_data.step_stream(1)
                    stat = fraud_data.get_stream_status()
                    last = stat.get("last_tx") or {}
                    fo_str = "🚨 Fan-out!" if last.get("is_fanout") else "1-hop"
                    s.write(f"Tx {stat['current_idx']}/{tot_tx}: `{last.get('from_account')}` ➔ `{last.get('to_account')}` ({last.get('amount_formatted')}) | {fo_str}")
                    time.sleep(1.0)
                s.update(label=f"✅ {steps} Transactions Ingested!", state="complete", expanded=False)
            st.rerun()

    btn_s3, btn_s4 = st.columns(2)
    with btn_s3:
        if st.button("⚡ Run All 100", key="btn_stream_all", use_container_width=True, disabled=(curr_tx >= tot_tx)):
            fraud_data.step_stream(tot_tx - curr_tx)
            st.rerun()
    with btn_s4:
        if st.button("🔄 Reset to 0", key="btn_stream_reset", use_container_width=True):
            fraud_data.reset_stream()
            st.session_state.selected_tx_id = None
            st.session_state.selected_sub_tx = None
            st.rerun()

    st.html("<br><hr><div style='text-align:center;color:#94a3b8;font-size:11px;'>© AML Fraud Investigation System</div>")

# ─── Top Header ───────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%I:%M:%S %p")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.html(textwrap.dedent("""
    <div>
        <h1 class="header-title">AML Fraud Detection</h1>
        <div class="header-subtitle">Graph Attention Network (GAT) AML Monitoring</div>
    </div>
    """))
with col_h2:
    st.html(textwrap.dedent(f"""
    <div style="text-align:right;margin-top:10px;">
        <span class="header-timestamp">🕒 Last Updated: {current_time}</span>
    </div>
    """))

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    total_txs_count = f"{len(fraud_data.get_transactions_df()):,}"
    flagged_senders = fraud_data.get_all_flagged_senders()
    high_count = sum(1 for s in flagged_senders if s["risk"] == "High")
    med_count = sum(1 for s in flagged_senders if s["risk"] == "Medium")
    low_count = sum(1 for s in flagged_senders if s["risk"] == "Low")

    stream_status = fraud_data.get_stream_status()
    curr_tx = stream_status["current_idx"]
    tot_tx = stream_status["total_txs"]
    pct = int((curr_tx / max(1, tot_tx)) * 100)
    rem = max(0, tot_tx - curr_tx)
    last_tx = stream_status.get("last_tx")

    st.html(textwrap.dedent(f"""
    <div class="top-banner">
        <div class="top-banner-icon">📋</div>
        <div>
            <div style="font-size:12px;color:#64748b;font-weight:600;">Streamed Transactions</div>
            <div class="banner-value">{curr_tx} / {tot_tx}</div>
            <div class="banner-subtext">{pct}% Processed · {rem} Remaining</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">High Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#dc2626;">{high_count}</div>
            <div style="font-size:12px;color:#dc2626;font-weight:600;">Require Review</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Medium Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#d97706;">{med_count}</div>
            <div style="font-size:12px;color:#d97706;font-weight:600;">Under Monitoring</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Low Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#16a34a;">{low_count}</div>
            <div style="font-size:12px;color:#16a34a;font-weight:600;">Low Priority</div>
        </div>
        <div style="margin-left:auto;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 18px;text-align:center;">
            <div style="font-size:11px;color:#16a34a;font-weight:700;">GAT MODEL</div>
            <div style="font-size:18px;font-weight:800;color:#16a34a;">● ACTIVE</div>
        </div>
    </div>
    """))

    # ── Live Stream Ingestion Notification & Pop-Up Ticker ──
    if not last_tx:
        st.html(f"""
        <div style="background:#f8fafc;border:2px dashed #94a3b8;border-radius:10px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:24px;">⏳</span>
                <div>
                    <div style="font-size:13.5px;font-weight:800;color:#1e293b;">Stream Engine Standing By (0 / {tot_tx} Transactions Streamed)</div>
                    <div style="font-size:12px;color:#64748b;margin-top:2px;">
                        Click <b>'▶ Step +1'</b>, <b>'⏩ Step +10'</b>, or <b>'⏱️ Stream 10 Live'</b> in the sidebar to stream transactions. Fan-out escalation and graph updates will appear here live.
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:10.5px;color:#64748b;font-weight:600;">GRAPH MEMORY</div>
                <div style="font-size:13px;font-weight:800;color:#0f172a;">0 Nodes · 0 Edges</div>
            </div>
        </div>
        """)
    else:
        lt_from = last_tx.get("from_account", "—")
        lt_to = last_tx.get("to_account", "—")
        lt_amt = last_tx.get("amount_formatted", "$0.00")
        lt_sig = str(last_tx.get("risk_tier", "Low")).upper()
        is_fo = last_tx.get("is_fanout", False)
        lt_fan = "⚠️ FAN-OUT DETECTED (≥2 RECEIVERS)" if is_fo else "NORMAL 1-HOP"
        lt_color = "#dc2626" if lt_sig == "HIGH" else "#d97706" if lt_sig == "MEDIUM" else "#16a34a"
        lt_bg = "#fef2f2" if lt_sig == "HIGH" else "#fffbeb" if lt_sig == "MEDIUM" else "#f0fdf4"
        lt_border = "#fecaca" if lt_sig == "HIGH" else "#fde68a" if lt_sig == "MEDIUM" else "#bbf7d0"

        # Pop-up toast for every new transaction
        if "last_seen_tx" not in st.session_state:
            st.session_state.last_seen_tx = None
        if last_tx.get("tx_id") != st.session_state.last_seen_tx:
            st.session_state.last_seen_tx = last_tx.get("tx_id")
            if is_fo:
                st.toast(f"🚨 Fan-Out Alert! TX {last_tx.get('tx_id')}: {lt_from} ➔ {lt_to} ({lt_amt})", icon="🚨")
            else:
                st.toast(f"⚡ Ingested TX {last_tx.get('tx_id')}: {lt_from} ➔ {lt_to} ({lt_amt})", icon="⚡")

        st.html(f"""
        <div style="background:{lt_bg};border:2px solid {lt_border};border-radius:10px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:24px;">⚡</span>
                <div>
                    <div style="font-size:13.5px;font-weight:800;color:#0f172a;">
                        Live Ingestion: <b>TX #{last_tx.get('tx_id')}</b> (Tx {curr_tx} of {tot_tx} · {pct}% Complete) · <span style="color:{lt_color};font-weight:800;">{lt_sig} RISK</span> · <span style="font-size:11px;font-weight:600;color:#64748b;">{lt_fan}</span>
                    </div>
                    <div style="font-size:12px;color:#334155;margin-top:3px;">
                        Sender: <b style="color:#1e40af;">{lt_from}</b> ➔ Receiver: <b style="color:#0f172a;">{lt_to}</b> | Amount: <b style="color:#059669;">{lt_amt}</b> | Format: <b>{last_tx.get('payment_format', 'Wire')}</b> | GAT Score: <b style="color:{lt_color};">{last_tx.get('risk_score', 0)}/100</b>
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:10.5px;color:#64748b;font-weight:600;">GRAPH MEMORY</div>
                <div style="font-size:13px;font-weight:800;color:#0f172a;">{stream_status.get('graph_nodes', 0)} Nodes · {stream_status.get('graph_edges', 0)} Edges</div>
            </div>
        </div>
        """)

        if is_fo:
            st.html(f"""
            <div style="background:#fef2f2;border:2px solid #ef4444;border-radius:8px;padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:22px;">🚨</span>
                <div>
                    <div style="font-size:13px;font-weight:800;color:#991b1b;">FAN-OUT PATTERN DETECTED!</div>
                    <div style="font-size:12px;color:#7f1d1d;margin-top:1px;">
                        Sender <b>{lt_from}</b> has now branched to <b>{last_tx.get('sender_unique_receivers', 2)} unique receivers</b>. Escalated to <b>{lt_sig} RISK</b> and added to Active Flagged Investigations!
                    </div>
                </div>
            </div>
            """)

    col_left, col_center, col_right = st.columns([1.0, 1.8, 1.2])

    # ════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Flagged Accounts (Fan-Out Groups)
    # ════════════════════════════════════════════════════════════════════
    with col_left:
        st.html('<div class="section-label">🚨 Flagged Sender Investigations</div>')

        all_txs = fraud_data.get_all_flagged_senders()

        if not all_txs:
            st.html("""
            <div style="background:#ffffff;border:1px dashed #cbd5e1;border-radius:10px;padding:24px 16px;text-align:center;">
                <div style="font-size:24px;margin-bottom:8px;">🟢</div>
                <div style="font-size:13px;font-weight:700;color:#334155;">No Active Alerts</div>
                <div style="font-size:11.5px;color:#64748b;margin-top:4px;">Stream transactions using the sidebar to monitor real-time graph escalation.</div>
            </div>
            """)
        else:
            if st.session_state.selected_tx_id not in [tx["tx_id"] for tx in all_txs]:
                st.session_state.selected_tx_id = all_txs[0]["tx_id"]

            for tx in all_txs:
                acc = tx.get("account", "Unknown")
                name = tx.get("name") or fraud_data.get_customer_profile(acc).get("name", acc)
                risk = tx.get("risk", "High")
                score = tx.get("risk_score", 100)
                pattern = tx.get("pattern", "FAN-OUT")
                gid = tx.get("group_id", 1)
                tx_id = tx.get("tx_id", f"GROUP-{gid}")
                is_sel = (tx_id == st.session_state.selected_tx_id or str(gid) == str(st.session_state.selected_tx_id))

                risk_color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#16a34a"
                sel_bg = "#eff6ff" if is_sel else "#ffffff"
                sel_border = "#93c5fd" if is_sel else "#e2e8f0"
                bar_width = max(5, min(100, score))
                tx_count_str = tx.get('tx_count', len(fraud_data.get_fan_out_rows(tx_id)))

                st.html(textwrap.dedent(f"""
                <div class="flagged-card flagged-card-{risk.lower()}"
                     style="background:{sel_bg}; border-color:{sel_border};">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div class="flagged-acc-id">Sender: {acc}</div>
                            <div style="font-size:11px;color:#475569;font-weight:500;margin-top:1px;">{name}</div>
                        </div>
                        <span class="badge-{risk.lower()}">{risk.upper()}</span>
                    </div>
                    <div class="flagged-pattern">📌 {pattern} · Group {gid} ({tx_count_str} txs)</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                        <div class="risk-score-bar-bg" style="flex:1;margin-right:8px;">
                            <div class="risk-score-bar-fill" style="width:{bar_width}%;background:{risk_color};"></div>
                        </div>
                        <span style="font-size:11px;font-weight:700;color:{risk_color};">{score}/100</span>
                    </div>
                </div>
                """))
                if st.button(f"Inspect Group {gid} →", key=f"left_btn_{tx_id}", use_container_width=True):
                    st.session_state.selected_tx_id = tx_id
                    st.session_state.selected_sub_tx = None
                    st.rerun()

    # ════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — Fan-Out Transaction Table + XAI + Human Decision
    # ════════════════════════════════════════════════════════════════════
    with col_center:
        if not all_txs:
            st.html("""
            <div class="center-card" style="text-align:center;padding:40px 20px;">
                <div style="font-size:36px;margin-bottom:12px;">🛡️</div>
                <div style="font-size:16px;font-weight:800;color:#0f172a;margin-bottom:6px;">Real-Time AML Monitoring Active</div>
                <div style="font-size:13px;color:#64748b;max-width:480px;margin:0 auto 12px auto;line-height:1.5;">
                    No high or medium risk fan-out clusters detected yet in current graph memory.
                </div>
                <div style="font-size:12px;color:#94a3b8;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;max-width:500px;margin:0 auto;">
                    💡 Use <b>'▶ Step +1'</b> or <b>'⏩ Step +10'</b> in the sidebar to stream transactions from <code>Data/testing_trans.csv</code>.<br>
                    Single sender transactions start as Low Risk; expanding to 2+ unique receivers triggers fan-out escalation.
                </div>
            </div>
            """)
        else:
            curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
            if not isinstance(curr_tx, dict) or not curr_tx:
                curr_tx = all_txs[0] if all_txs else {}
                if isinstance(curr_tx, dict):
                    st.session_state.selected_tx_id = curr_tx.get("group_id") or curr_tx.get("tx_id")
            
            risk = curr_tx.get("risk", "High")
            risk_score = curr_tx.get("risk_score", 100)
            risk_col = "#dc2626" if risk == "High" else "#d97706" if risk == "Medium" else "#16a34a"
            pattern = curr_tx.get("pattern", "FAN-OUT")
            gid = curr_tx.get("group_id", 1)

            lead_tx_id = curr_tx.get('lead_tx_id', curr_tx.get('tx_id', f'GROUP-{gid}'))
            sender_acc = curr_tx.get('account', '—')
            sender_name = curr_tx.get('name') or fraud_data.get_customer_profile(sender_acc).get('name', sender_acc)
            payment_format = curr_tx.get('payment_format', '—')
            first_ts = curr_tx.get('initial_timestamp', curr_tx.get('timestamp', '—'))
            latest_ts = curr_tx.get('latest_timestamp', curr_tx.get('timestamp', '—'))
            ts_display = f"Burst Start: {first_ts} · Latest: {latest_ts}" if first_ts != latest_ts else f"Timestamp: {first_ts}"

            st.html(textwrap.dedent(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <div>
                    <div style="font-size:16px;font-weight:800;color:#0f172a;">
                        Fan-out Group {gid} · TX {lead_tx_id}
                        <span style="font-size:12px;font-weight:500;color:#64748b;margin-left:6px;">{pattern}</span>
                    </div>
                    <div style="font-size:12px;color:#475569;margin-top:2px;">
                        Sender: <b>{sender_acc}</b> ({sender_name}) ·
                        {ts_display} · {payment_format}
                    </div>
                </div>
                <span class="badge-{risk.lower()}">{risk.upper()} RISK · {risk_score}/100</span>
            </div>
            """))

            if st.button("🕸️  Tap to View as Graph Network", key="graph_btn", use_container_width=False):
                st.session_state.goto_graph = True
                st.rerun()

            st.html('<div class="section-label">Transaction Routing Flow (Source & Receivers)</div>')

            fan_rows = fraud_data.get_fan_out_rows(st.session_state.selected_tx_id, include_source=True)
            if fan_rows:
                df_fan = pd.DataFrame(fan_rows)
                df_display = df_fan[["role", "account", "to_entity_name", "amount", "time", "payment_format", "gat_signal"]].copy()
                df_display.columns = ["Role", "Account Number", "Entity Name", "Amount", "Timestamp", "Payment Format", "GAT Signal"]
            else:
                df_display = pd.DataFrame(columns=["Role", "Account Number", "Entity Name", "Amount", "Timestamp", "Payment Format", "GAT Signal"])

            event = st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="fan_table"
            )

            selected_rows = event.selection.get("rows", []) if event.selection else []
            if selected_rows and len(fan_rows) > selected_rows[0]:
                row_idx = selected_rows[0]
                sub_tx_data = fan_rows[row_idx]
                st.session_state.selected_sub_tx = sub_tx_data

            st.html("""
            <div style="font-size:10.5px;color:#94a3b8;margin-top:4px;">
                🖱️ Click any row (Source Sender or Receiver) to view its customer profile in the right panel.
            </div>
            """)

            is_high = (risk == "High")
            xai_extra = "xai-box-high" if is_high else ""
            exps_html = "".join([f"<li>{e}</li>" for e in curr_tx.get("explanations", [])])
            fraud_icon = "⚠️" if curr_tx.get("is_fraud", False) else "ℹ️"
            fraud_label = "FRAUD DETECTED" if curr_tx.get("is_fraud", False) else "SUSPICIOUS ACTIVITY"

            st.html(textwrap.dedent(f"""
            <div class="xai-box {xai_extra}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <div class="xai-title">{fraud_icon} {fraud_label} — Why was this flagged?</div>
                    <span class="badge-{risk.lower()}">{pattern}</span>
                </div>
                <ul class="xai-list">
                    {exps_html}
                </ul>
                <div style="font-size:11px;color:#64748b;margin-top:10px;border-top:1px solid #fde68a;padding-top:8px;">
                    Model: <b>{curr_tx.get('model_used', 'GAT AML Model (PyG)')}</b> &nbsp;|&nbsp;
                    Confidence: <b>{curr_tx.get('model_confidence', f'{risk_score}%')}</b> &nbsp;|&nbsp;
                    Risk Score: <b style="color:{risk_col};">{risk_score}/100</b> (Raw GAT Sigmoid: <code>{curr_tx.get('gat_prob', 0):.6f}</code>)
                </div>
            </div>
            """))

            # Authorised Bank Auditor Decision Panel
            if is_high or risk == "Medium":
                tx_key = curr_tx.get("tx_id", f"GROUP-{gid}")
                already_submitted = st.session_state.human_decision_submitted.get(tx_key)

                st.html(textwrap.dedent(f"""
                <div class="human-decision-panel">
                    <div class="human-decision-title">
                        🏦 Authorised Bank Auditor Decision Required — Group {gid}
                    </div>
                    <div style="font-size:12px;color:#92400e;margin-bottom:12px;">
                        Risk Score: <b>{risk_score}/100</b> — This Fan-Out investigation requires an authorised bank auditor decision.
                    </div>
                </div>
                """))

                if already_submitted:
                    decision_val = already_submitted["decision"]
                    st.success(f"**Decision Recorded:** {decision_val}")
                    st.info(f"**Authorised Bank Auditor Notes:** {already_submitted['notes'] or '(none)'}")
                    if st.button("Revise Decision", key=f"revise_{tx_key}"):
                        del st.session_state.human_decision_submitted[tx_key]
                        st.rerun()
                else:
                    dec_col1, dec_col2 = st.columns([1, 1])
                    with dec_col1:
                        decision = st.radio(
                            "**Authorised Bank Auditor Decision**",
                            ["✅ Approve Transaction", "🚫 Block Transaction", "📤 Escalate to Senior Investigator"],
                            key=f"decision_{tx_key}",
                            index=2
                        )
                    with dec_col2:
                        notes = st.text_area(
                            "**Authorised Bank Auditor Notes**",
                            placeholder="Add reasoning, observations, or escalation notes...",
                            key=f"notes_{tx_key}",
                            height=180
                        )

                    st.html("""
                    <div class="human-decision-disclaimer">
                        ⚠️ <b>Disclaimer:</b> By submitting, you confirm this decision is made by an
                        AUTHORISED BANK AUDITOR. The GAT AML model provided supporting analysis only.
                        This action will be logged and audited.
                    </div>
                    """)

                    if st.button(f"📋 Submit Decision for Group {gid}", key=f"submit_{tx_key}", type="primary", use_container_width=True):
                        fraud_data.submit_auditor_decision(tx_key, decision, notes)
                        st.session_state.human_decision_submitted[tx_key] = {
                            "decision": decision,
                            "notes": notes,
                            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
                        }
                        if "Approve" in decision:
                            rem_txs = fraud_data.get_all_flagged_senders()
                            st.session_state.selected_tx_id = rem_txs[0]["tx_id"] if rem_txs else None
                            st.session_state.selected_sub_tx = None
                        st.rerun()

            st.html(textwrap.dedent("""
            <div class="ai-advisory" style="margin-top:16px;">
                <span style="font-size:16px;">🤖</span>
                <span>
                    <b>AI Advisory:</b> The analysis above is generated by the GAT AML model to
                    <b>support the fraud investigator's decision</b>. The AI does not block or approve
                    transactions. All final decisions must be made by an AUTHORIZED BANK AUDITOR.
                </span>
            </div>
            """))

    # ════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Customer Profile (Source or Receiver)
    # ════════════════════════════════════════════════════════════════════
    with col_right:
        if not all_txs:
            st.html("""
            <div class="profile-card profile-empty">
                <div style="font-size:32px;margin-bottom:10px;">👤</div>
                <div style="font-weight:700;font-size:14px;color:#475569;">No Account Selected</div>
                <div style="font-size:11.5px;color:#94a3b8;margin-top:4px;">Stream transactions to view real-time customer behavioral profiles.</div>
            </div>
            """)
        else:
            sub_tx = st.session_state.selected_sub_tx

            if sub_tx is None:
                source_acc = curr_tx.get("account", "—")
                sub_tx = {
                    "account": source_acc,
                    "to_account": source_acc,
                    "is_source": True,
                    "to_entity_name": curr_tx.get("name", "—"),
                    "to_bank_name": curr_tx.get("bank_name", "—"),
                    "to_bank_id": curr_tx.get("bank_id", "—"),
                    "to_entity_id": curr_tx.get("entity_id", "—"),
                    "payment_format": curr_tx.get("payment_format", "Wire"),
                    "gat_signal": curr_tx.get("gat_signal", "HIGH"),
                    "amount": curr_tx.get("amount_formatted", "—"),
                }

            acc_num = sub_tx.get("account") or sub_tx.get("to_account", "—")
            is_src = sub_tx.get("is_source", False)
            profile_label = "Source Sender Profile" if is_src else "Receiver Profile"

            try:
                profile = fraud_data.get_customer_profile(acc_num)
            except Exception:
                profile = {}

            if not isinstance(profile, dict):
                profile = {}

            profile_name = profile.get("name") if profile.get("name") not in ("—", "Unknown", None, "") else sub_tx.get("to_entity_name", acc_num)
            bank_name = profile.get("bank_name") if profile.get("bank_name") not in ("—", None, "") else sub_tx.get("to_bank_name", "—")
            bank_id = profile.get("bank_id") if profile.get("bank_id") not in ("—", None, "") else sub_tx.get("to_bank_id", "—")
            entity_id = profile.get("entity_id") if profile.get("entity_id") not in ("—", None, "") else sub_tx.get("to_entity_id", "—")
            tot_inc = profile.get("total_incoming") if profile.get("total_incoming") not in ("—", None, "") else sub_tx.get("amount", "—")
            tot_out = profile.get("total_outgoing", "—")
            avg_in = profile.get("avg_incoming_amount", "—")
            avg_out = profile.get("avg_outgoing_amount", "—")
            max_in = profile.get("max_incoming_amount", "—")
            max_out = profile.get("max_outgoing_amount", "—")
            in_tx = profile.get("previous_incoming", sub_tx.get("previous_incoming", "—"))
            out_tx = profile.get("previous_outgoing", sub_tx.get("previous_outgoing", "—"))
            uniq_snds = profile.get("unique_senders", "—")
            uniq_recs = profile.get("unique_receivers", "—")
            tot_deg = profile.get("total_degree", "—")
            net_flow = profile.get("net_flow", "—")

            risk_tier = profile.get("risk_tier", "High Risk" if "HIGH" in str(sub_tx.get("gat_signal", "HIGH")).upper() else ("Medium Risk" if "MEDIUM" in str(sub_tx.get("gat_signal", "")).upper() else "Low Risk"))
            tier_color = "#dc2626" if "High" in risk_tier else "#d97706" if "Medium" in risk_tier else "#16a34a"
            payment_fmt = str(sub_tx.get('payment_format', '—'))

            beh_rows = [
                ("Total Incoming Amount", tot_inc),
                ("Average Incoming Amount", avg_in),
                ("Maximum Incoming Amount", max_in),
                ("Incoming Transactions", str(in_tx)),
                ("Unique Senders", str(uniq_senders) if (uniq_senders := uniq_snds) else str(uniq_snds)),
                ("Total Outgoing Amount", tot_out),
                ("Average Outgoing Amount", avg_out),
                ("Maximum Outgoing Amount", max_out),
                ("Outgoing Transactions", str(out_tx)),
                ("Unique Receivers", str(uniq_recs)),
                ("Total Degree", str(tot_deg)),
                ("Net Flow", net_flow),
            ]
            beh_rows_html = "".join([f"<tr><td><b>{r[0]}</b></td><td style='text-align:right;'>{r[1]}</td></tr>" for r in beh_rows if r[1] != "—" and r[1] != ""])

            st.html(textwrap.dedent(f"""
            <div class="profile-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div>
                        <div class="section-label">{profile_label}</div>
                        <div style="font-size:16px;font-weight:800;color:#0f172a;">{profile_name}</div>
                        <div style="font-size:12px;color:#64748b;margin-top:2px;">Account: <b>{acc_num}</b></div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{tier_color}20;color:{tier_color};border:1px solid {tier_color}44;
                                     padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;">
                            {risk_tier}
                        </span>
                    </div>
                </div>

                <div class="profile-metric-grid">
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Bank Name</div>
                        <div class="profile-metric-val" style="font-size:12px;">{bank_name}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Bank ID</div>
                        <div class="profile-metric-val">{bank_id}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Entity ID</div>
                        <div class="profile-metric-val">{entity_id}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Payment Format</div>
                        <div class="profile-metric-val" style="font-size:12px;">{payment_fmt}</div>
                    </div>
                </div>

                <div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:6px;">Financial Summary</div>
                <table class="custom-table">
                    <tbody>
                        {beh_rows_html}
                    </tbody>
                </table>
            </div>
            """))


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NETWORK PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Graph Network":
    st.subheader("🕸️ Money Trail — Graph Network Topology")
    st.markdown(
        "Visualizing transactional connections for the selected Fan-Out group. "
        "Hover over any node to see the customer profile. "
        "Sender (★) → Hop-1 Receivers (●)"
    )

    all_groups = fraud_data.get_all_flagged_senders()
    if not all_groups:
        st.info("No active fan-out investigations in graph memory yet. Stream transactions from the sidebar to visualize the money trail.")
    else:
        group_options = [g["tx_id"] for g in all_groups]
        default_sel = st.session_state.selected_tx_id
        default_idx = group_options.index(default_sel) if default_sel in group_options else 0

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            sel_tx = st.selectbox(
                "Select Fan-out Group Network to Inspect",
                group_options,
                index=default_idx
            )
        with col_g2:
            show_2hop = st.checkbox("Show 2-Hop Neighbors", value=False, disabled=True, help="Hop-2 nodes are derived only when complete downstream data is available in the dataset.")

        tx_info = fraud_data.get_transaction_by_id(sel_tx)
        if not isinstance(tx_info, dict) or not tx_info:
            tx_info = all_groups[0] if all_groups else {}

        risk_col = "#dc2626" if tx_info.get("risk") == "High" else "#d97706" if tx_info.get("risk") == "Medium" else "#16a34a"
        st.html(textwrap.dedent(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:14px 18px;margin-bottom:16px;display:flex;gap:28px;align-items:center;">
            <div>
                <div style="font-size:11px;color:#64748b;font-weight:600;">PATTERN</div>
                <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('pattern', 'FAN-OUT')}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;font-weight:600;">GAT RISK SCORE</div>
                <div style="font-size:14px;font-weight:700;color:{risk_col};">{tx_info.get('risk_score', 100)}/100</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;font-weight:600;">MODEL</div>
                <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('model_used', 'GAT AML Model')}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;font-weight:600;">SENDER</div>
                <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('account', '—')} ({tx_info.get('name') or fraud_data.get_customer_profile(tx_info.get('account', '')).get('name', '—')})</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;font-weight:600;">TOTAL AMOUNT</div>
                <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('amount_formatted', '—')}</div>
            </div>
        </div>
        """))

        fig_g = graph_vis.render_plotly_graph(sel_tx, include_2hop=False)
        st.plotly_chart(fig_g, use_container_width=True)

        st.html(textwrap.dedent("""
        <div style="display:flex;gap:24px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
                <span style="width:14px;height:14px;background:#ef4444;border-radius:50%;display:inline-block;"></span>
                Sender Account
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
                <span style="width:14px;height:14px;background:#f59e0b;border-radius:50%;display:inline-block;"></span>
                Hop-1 Direct Receivers
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
                <span style="width:28px;height:2px;background:#ef4444;display:inline-block;"></span>
                Hop-1 Transfer (amount shown)
            </div>
        </div>
        """))

        st.markdown("---")
        st.html(textwrap.dedent(f"""
        <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:14px 16px;">
            <div style="font-weight:700;font-size:13px;color:#dc2626;margin-bottom:8px;">⚠️ Why was this flagged?</div>
            <ul style="margin:0;padding-left:18px;font-size:12px;color:#334155;line-height:1.8;">
                {"".join([f"<li>{e}</li>" for e in tx_info.get("explanations", [])])}
            </ul>
        </div>
        """))



