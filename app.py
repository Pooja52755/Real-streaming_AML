import os
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

def format_probability(p):
    if p is None:
        return "0.000000"
    try:
        p_val = float(p)
    except (ValueError, TypeError):
        return str(p)
    if p_val == 0.0:
        return "0.000000"
    if p_val < 0.0001:
        return f"{p_val:.8f}"
    return f"{p_val:.6f}"



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
    .badge-approved {
        background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-blocked {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-revised {
        background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe;
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
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False
if "stream_delay" not in st.session_state:
    st.session_state.stream_delay = 1.0
if "revision_target_id" not in st.session_state:
    st.session_state.revision_target_id = None
if "selected_pred_tx_id" not in st.session_state:
    st.session_state.selected_pred_tx_id = None
if "retrain_result" not in st.session_state:
    st.session_state.retrain_result = None

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
            <div class="sidebar-brand-subtitle">GAT Transaction Prediction & Audit</div>
        </div>
    </div>
    """))

    nav_options = [
        "Dashboard",
        "Reviewed Transactions",
        "Decisions",
        "Graph Network"
    ]
    default_page_idx = 3 if st.session_state.goto_graph else 0
    page = st.radio(
        "Navigation",
        nav_options,
        index=default_page_idx,
        label_visibility="collapsed"
    )
    if st.session_state.goto_graph and page == "Graph Network":
        st.session_state.goto_graph = False

    # ── Streaming Simulation Controls ──
    st.markdown("---")
    st.markdown("<div style='font-size:12px;font-weight:700;color:#0f172a;margin-bottom:4px;'>⚡ Transaction Streaming Engine</div>", unsafe_allow_html=True)
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

    # Main Stream Data continuous playback button
    if not st.session_state.is_streaming:
        if st.button("▶️ Stream Data", key="btn_stream_auto", type="primary", use_container_width=True, disabled=(curr_tx >= tot_tx)):
            st.session_state.is_streaming = True
            fraud_data.step_stream(1)
            st.rerun()
    else:
        if st.button("⏸️ Stop Streaming", key="btn_stream_stop", type="secondary", use_container_width=True):
            st.session_state.is_streaming = False
            st.rerun()

    # Manual Single & Multi-step buttons
    btn_s1, btn_s2 = st.columns(2)
    with btn_s1:
        if st.button("▶ Step +1", key="btn_stream_1", use_container_width=True, disabled=(curr_tx >= tot_tx or st.session_state.is_streaming)):
            fraud_data.step_stream(1)
            st.rerun()
    with btn_s2:
        if st.button("⏩ Step +10", key="btn_stream_10", use_container_width=True, disabled=(curr_tx >= tot_tx or st.session_state.is_streaming)):
            fraud_data.step_stream(10)
            st.rerun()

    # Streaming Speed selector
    st.session_state.stream_delay = st.select_slider(
        "Stream Speed",
        options=[0.5, 1.0, 2.0],
        value=st.session_state.stream_delay,
        format_func=lambda s: f"{s}s / tx",
        disabled=st.session_state.is_streaming
    )

    btn_s3, btn_s4 = st.columns(2)
    with btn_s3:
        if st.button(f"⚡ Run All ({tot_tx})", key="btn_stream_all", use_container_width=True, disabled=(curr_tx >= tot_tx or st.session_state.is_streaming)):
            fraud_data.step_stream(tot_tx - curr_tx)
            st.rerun()
    with btn_s4:
        if st.button("🔄 Reset to 0", key="btn_stream_reset", use_container_width=True):
            st.session_state.is_streaming = False
            fraud_data.reset_stream()
            st.session_state.selected_tx_id = None
            st.session_state.selected_sub_tx = None
            st.rerun()

    # ── Reviewed Transactions Filter & Feedback Options on Sidebar ──
    st.markdown("---")
    st.markdown("<div style='font-size:12px;font-weight:800;color:#0f172a;margin-bottom:6px;'>⚖️ Reviewed Transactions</div>", unsafe_allow_html=True)
    all_decs_sidebar = fraud_data.get_all_auditor_decisions()
    app_count = sum(1 for d in all_decs_sidebar if d.get("decision") == "Approve")
    rej_count = sum(1 for d in all_decs_sidebar if d.get("decision") == "Reject")
    esc_count = sum(1 for d in all_decs_sidebar if d.get("decision") == "Escalate")

    sidebar_decision_view = st.radio(
        "Reviewed Transactions:",
        [
            f"🚨 Active Alerts ({active_cnt})",
            f"✅ Approved Cases ({app_count})",
            f"🚫 Rejected Cases ({rej_count})",
            f"⚠️ Escalated Cases ({esc_count})",
            f"📋 All Decisions Ledger ({len(all_decs_sidebar)})"
        ],
        key="sidebar_decision_view"
    )
    st.caption("💡 Filter groups by compliance status: **Approved**, **Rejected**, or **Escalated**.")

    sb_stat = fraud_data.get_supabase_status()
    sb_conn = sb_stat.get("connected", False)
    if sb_conn:
        st.markdown(
            "<div style='background:#ecfdf5;border:1px solid #10b981;border-radius:6px;padding:8px 10px;margin-top:12px;font-size:11px;font-weight:700;color:#065f46;text-align:center;'>"
            "🟢 Supabase PostgreSQL: CONNECTED"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#f8fafc;border:1px dashed #94a3b8;border-radius:6px;padding:8px 10px;margin-top:12px;font-size:11px;color:#64748b;text-align:center;'>"
            "⚪ Supabase: Standby (Secrets Needed)"
            "</div>",
            unsafe_allow_html=True
        )

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

# ─── Supabase Cloud Status Indicator & Popup ──────────────────────────────
sb_status = fraud_data.get_supabase_status()
sb_connected = sb_status.get("connected", False)
tables_ready = sb_status.get("tables_ready", False)

if sb_connected:
    if "sb_conn_toast_shown" not in st.session_state:
        if tables_ready:
            st.toast("⚡ Supabase Cloud Connected! PostgreSQL Database Active.", icon="🟢")
        else:
            st.toast("⚡ Supabase Connected! Ready for tables setup.", icon="🟢")
        st.session_state.sb_conn_toast_shown = True

    if tables_ready:
        st.html("""
        <div style="background: linear-gradient(90deg, #ecfdf5, #f0fdf4); border: 1.5px solid #10b981; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 26px;">🟢</span>
                <div>
                    <div style="font-size: 14px; font-weight: 800; color: #065f46;">Supabase Cloud Connected — Live PostgreSQL Active</div>
                    <div style="font-size: 11.5px; color: #047857; margin-top: 1px;">
                        Persistent Source of Truth: 100 Transactions, Graph Topology & Human Authorizer Decisions are stored live in Supabase PostgreSQL.
                    </div>
                </div>
            </div>
            <span style="background: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em;">LIVE DATABASE</span>
        </div>
        """)
    else:
        st.html("""
        <div style="background: linear-gradient(90deg, #eff6ff, #f8fafc); border: 1.5px solid #3b82f6; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.12);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 26px;">🟢</span>
                <div>
                    <div style="font-size: 14px; font-weight: 800; color: #1e40af;">Supabase Cloud Connected (API Active)</div>
                    <div style="font-size: 11.5px; color: #2563eb; margin-top: 1px;">
                        Connected to project <code>jqojlzkxntkekbkjiijq</code>. To initialize PostgreSQL tables without CSVs, run <code>supabase_schema.sql</code> once in your Supabase SQL Editor.
                    </div>
                </div>
            </div>
            <span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em;">CONNECTED</span>
        </div>
        """)
else:
    st.html("""
    <div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 24px;">⚠️</span>
            <div>
                <div style="font-size: 13.5px; font-weight: 800; color: #92400e;">Supabase Cloud: Standby Mode</div>
                <div style="font-size: 11.5px; color: #b45309; margin-top: 1px;">
                    Connect to Supabase by configuring <code>SUPABASE_URL</code> and <code>SUPABASE_KEY</code> in Streamlit Cloud Secrets.
                </div>
            </div>
        </div>
        <span style="background: #f59e0b; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10.5px; font-weight: 700;">STANDBY</span>
    </div>
    """)

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
        # Check sidebar filter
        side_view = st.session_state.get("sidebar_decision_view", "Active Alerts")
        if "Approved" in side_view:
            sec_title = "✅ Approved Groups"
            all_txs = fraud_data.get_approved_investigations()
            empty_msg = "No groups have been marked as Approved yet. Review active alerts to approve legitimate accounts."
        elif "Rejected" in side_view:
            sec_title = "🚫 Rejected Groups"
            all_txs = fraud_data.get_rejected_investigations()
            empty_msg = "No groups have been marked as Rejected yet. Review active alerts to reject suspicious accounts."
        elif "Escalated" in side_view:
            sec_title = "⚠️ Escalated Groups (SAR Review)"
            all_txs = fraud_data.get_escalated_investigations()
            empty_msg = "No groups have been escalated yet. Review active alerts to escalate complex cases."
        elif "All Decisions" in side_view:
            sec_title = "📋 All Audited Groups"
            all_txs = [inv for inv in fraud_data.get_all_investigations() if inv.get("auditor_decision") in ["Approve", "Reject", "Escalate"]]
            empty_msg = "No authorizer decisions have been recorded yet."
        else:
            sec_title = "🚨 Active Flagged Investigations"
            all_txs = fraud_data.get_all_flagged_senders()
            empty_msg = "No active alerts. Stream transactions using the sidebar to monitor graph escalation."

        st.html(f'<div class="section-label">{sec_title}</div>')

        if not all_txs:
            st.html(f"""
            <div style="background:#ffffff;border:1px dashed #cbd5e1;border-radius:10px;padding:24px 16px;text-align:center;">
                <div style="font-size:24px;margin-bottom:8px;">ℹ️</div>
                <div style="font-size:13px;font-weight:700;color:#334155;">{empty_msg}</div>
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
                if score >= 80:
                    risk = "High"
                pattern = tx.get("pattern", "FAN-OUT")
                gid = tx.get("group_id", 1)
                tx_id = tx.get("tx_id", f"GROUP-{gid}")
                is_sel = (tx_id == st.session_state.selected_tx_id or str(gid) == str(st.session_state.selected_tx_id))

                aud_decision = tx.get("auditor_decision")
                if aud_decision == "Approve":
                    dec_pill = '<span style="background:#dcfce7;color:#15803d;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">✅ APPROVED</span>'
                elif aud_decision == "Reject":
                    dec_pill = '<span style="background:#fee2e2;color:#b91c1c;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">🚫 REJECTED</span>'
                elif aud_decision == "Escalate":
                    dec_pill = '<span style="background:#fef3c7;color:#b45309;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">⚠️ ESCALATED</span>'
                else:
                    dec_pill = f'<span class="badge-{risk.lower()}">{risk.upper()}</span>'

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
                        {dec_pill}
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
            pass
        else:
            curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
            if not isinstance(curr_tx, dict) or not curr_tx:
                curr_tx = all_txs[0] if all_txs else {}
                if isinstance(curr_tx, dict):
                    st.session_state.selected_tx_id = curr_tx.get("group_id") or curr_tx.get("tx_id")
            
            risk = curr_tx.get("risk", "High")
            risk_score = curr_tx.get("risk_score", 100)
            if risk_score >= 80:
                risk = "High"
            risk_col = "#dc2626" if risk == "High" else "#d97706" if risk == "Medium" else "#16a34a"
            pattern = curr_tx.get("pattern", "FAN-OUT")
            gid = curr_tx.get("group_id", 1)

            lead_tx_id = curr_tx.get('lead_tx_id', curr_tx.get('tx_id', f'GROUP-{gid}'))
            sender_acc = curr_tx.get('account', '—')
            sender_name = curr_tx.get('name') or fraud_data.get_customer_profile(sender_acc).get('name', sender_acc)
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
                        Sender: <b>{sender_acc}</b> ({sender_name}) · {ts_display}
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
            exps_clean = []
            gat_prob_val = curr_tx.get("gat_prob", 0.0)
            gat_prob_str = format_probability(gat_prob_val)
            for e in curr_tx.get("explanations", []):
                if "payment format" in e.lower() or "currency:" in e.lower():
                    continue
                if "risk probability" in e.lower() or "gat graph" in e.lower() or "neural network" in e.lower() or "ensemble" in e.lower():
                    exps_clean.append(f"GAT Graph Attention Network (PyG): {risk.upper()} (Risk Score: {risk_score}/100)")
                else:
                    exps_clean.append(e)
            exps_html = "".join([f"<li>{e}</li>" for e in exps_clean])
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
                    Model: <b>{curr_tx.get('model_used', 'PyTorch Geometric GAT AML Model')}</b> &nbsp;|&nbsp;
                    Confidence: <b>{curr_tx.get('model_confidence', f'{risk_score}%')}</b> &nbsp;|&nbsp;
                    Risk Score: <b style="color:{risk_col};">{risk_score}/100</b> &nbsp;|&nbsp;
                    Status: <b style="color:{risk_col};">{risk.upper()} RISK ALERT</b>
                </div>
            </div>
            """))

            # Authorised Bank Authorizer Decision Panel
            tx_key = curr_tx.get("tx_id", f"GROUP-{gid}")
            all_decs_lookup = {d["target_id"]: d for d in fraud_data.get_all_auditor_decisions()}
            already_decided = all_decs_lookup.get(tx_key) or all_decs_lookup.get(f"GROUP-{curr_tx.get('account')}")

            if already_decided:
                dec_val = already_decided.get("decision", "Approve")
                is_app = (dec_val == "Approve")
                is_esc = (dec_val == "Escalate")
                b_color = "#10b981" if is_app else ("#f59e0b" if is_esc else "#ef4444")
                bg_color = "#f0fdf4" if is_app else ("#fffbeb" if is_esc else "#fef2f2")
                txt_color = "#166534" if is_app else ("#b45309" if is_esc else "#991b1b")
                verdict_title = "✅ AUTHORIZER DECISION: APPROVE" if is_app else ("⚠️ AUTHORIZER DECISION: ESCALATE (SAR)" if is_esc else "🚫 AUTHORIZER DECISION: REJECT")

                st.html(textwrap.dedent(f"""
                <div style="background:{bg_color};border:2px solid {b_color};border-radius:10px;padding:16px 20px;margin-top:14px;margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:15px;font-weight:900;color:{txt_color};">
                                {verdict_title}
                            </span>
                        </div>
                        <span style="font-size:11.5px;color:#64748b;">🕒 {already_decided.get('timestamp', '—')}</span>
                    </div>
                    <div style="font-size:12.5px;color:#0f172a;margin-top:10px;background:#ffffff;padding:10px 14px;border-radius:6px;border-left:4px solid {b_color};">
                        <b>Authorizer Feedback Notes:</b> {already_decided.get('notes') or '(No compliance notes entered)'}
                    </div>
                    {f'<div style="font-size:11.5px;color:#2563eb;margin-top:6px;"><b>Revision Remarks:</b> {already_decided.get("revision_remark")}</div>' if already_decided.get("is_revised") and already_decided.get("revision_remark") else ''}
                </div>
                """))

                rev_active_key = f"dash_rev_active_{tx_key}"
                if not st.session_state.get(rev_active_key, False):
                    if st.button(f"✏️ Revise Decision for Group {gid}", key=f"btn_open_rev_{tx_key}", use_container_width=True):
                        st.session_state[rev_active_key] = True
                        st.rerun()
                else:
                    st.html("""
                    <div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                        <div style="font-size:13px;font-weight:800;color:#92400e;">✏️ Revise Authorizer Decision & Feedback</div>
                        <div style="font-size:11.5px;color:#78350f;">Update the verdict or compliance notes. Audit trail will be preserved.</div>
                    </div>
                    """)
                    r_c1, r_c2 = st.columns([1, 1.3])
                    with r_c1:
                        rev_idx = 0 if is_app else (2 if is_esc else 1)
                        new_choice = st.radio(
                            "Revised Verdict",
                            ["Approve", "Reject", "Escalate"],
                            index=rev_idx,
                            format_func=lambda c: "✅ Approve (Legitimate)" if c == "Approve" else ("🚫 Reject (Confirmed Laundering)" if c == "Reject" else "⚠️ Escalate (Senior Review / SAR)"),
                            key=f"radio_rev_dash_{tx_key}"
                        )
                    with r_c2:
                        new_notes = st.text_area(
                            "Updated Feedback Notes",
                            value=already_decided.get("notes", ""),
                            key=f"area_rev_dash_{tx_key}",
                            height=90
                        )
                    rev_remark = st.text_input("Reason for Revision", value="Authorizer audit re-assessment", key=f"inp_remark_dash_{tx_key}")
                    b_r1, b_r2 = st.columns([1, 1])
                    with b_r1:
                        if st.button("💾 Save Revised Decision", key=f"btn_save_rev_dash_{tx_key}", type="primary", use_container_width=True):
                            fraud_data.revise_auditor_decision(tx_key, new_choice, new_notes, rev_remark)
                            st.session_state[rev_active_key] = False
                            st.toast(f"Decision for Group {gid} updated to {new_choice}!", icon="✅")
                            st.rerun()
                    with b_r2:
                        if st.button("Cancel", key=f"btn_cancel_rev_dash_{tx_key}", use_container_width=True):
                            st.session_state[rev_active_key] = False
                            st.rerun()

            elif is_high or risk == "Medium":
                st.html(textwrap.dedent(f"""
                <div class="human-decision-panel">
                    <div class="human-decision-title">
                        🏦 Authorised Bank Authorizer Decision Required — Group {gid}
                    </div>
                    <div style="font-size:12px;color:#92400e;margin-bottom:8px;">
                        Risk Score: <b>{risk_score}/100</b> — Review this flagged fan-out group and record an authoritative compliance verdict.
                    </div>
                </div>
                """))

                dec_col1, dec_col2 = st.columns([1, 1.2])
                with dec_col1:
                    decision = st.radio(
                        "**Authorizer Decision**",
                        ["Approve", "Reject", "Escalate"],
                        format_func=lambda c: "✅ Approve (Legitimate)" if c == "Approve" else ("🚫 Reject (Confirmed Laundering)" if c == "Reject" else "⚠️ Escalate (Senior Review / SAR)"),
                        key=f"decision_{tx_key}",
                        index=1 if risk in ["High", "Medium"] else 0
                    )
                with dec_col2:
                    notes = st.text_area(
                        "**Authorizer Feedback Notes**",
                        placeholder="Add compliance notes, counterparty verification, or fan-out justification...",
                        key=f"notes_{tx_key}",
                        height=130
                    )

                if st.button(f"📋 Submit Decision for Group {gid}", key=f"submit_{tx_key}", type="primary", use_container_width=True):
                    fraud_data.submit_auditor_decision(tx_key, decision, notes, target_type="group")
                    st.session_state.selected_tx_id = None
                    st.session_state.selected_sub_tx = None
                    st.toast(f"Decision for Group {gid} saved ({decision})! Alert resolved and moved to Reviewed Transactions.", icon="✅")
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
                <div style="font-size:11.5px;color:#94a3b8;margin-top:4px;">Stream transactions to view customer behavioral profiles.</div>
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
#  SCREEN 2: PREDICTIONS & HUMAN FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
elif page in ["Reviewed Transactions", "Screen 2: Predictions & Feedback"]:
    st.html(textwrap.dedent("""
    <div>
        <h2 style="font-size:22px;font-weight:800;color:#0f172a;margin:0;">Reviewed Transactions</h2>
        <div style="font-size:13px;color:#64748b;margin-top:2px;">
            Full compliance record of all transactions and fan-out groups reviewed and adjudicated by the human authorizer. Submissions are synced persistently to Supabase PostgreSQL.
        </div>
    </div>
    <br>
    """))

    proc_txs = fraud_data.get_processed_transactions_list()
    all_decisions_list = fraud_data.get_all_auditor_decisions()
    all_decisions = {d["target_id"]: d for d in all_decisions_list}

    def get_decision_for_tx(tx_id, from_account):
        if tx_id in all_decisions:
            return all_decisions[tx_id]
        if f"GROUP-{from_account}" in all_decisions:
            return all_decisions[f"GROUP-{from_account}"]
        if from_account in all_decisions:
            return all_decisions[from_account]
        return None

    tot_reviewed = len(all_decisions_list)

    # Top Metric Banner (Only Reviewed count kept as requested)
    col_k1, _ = st.columns([1, 3])
    with col_k1:
        st.metric("Reviewed Transactions", f"{tot_reviewed}")

    if tot_reviewed == 0:
        st.html("""
        <div style="background:#ffffff;border:1.5px dashed #cbd5e1;border-radius:12px;padding:48px 24px;text-align:center;margin-top:16px;">
            <div style="font-size:42px;margin-bottom:12px;">🛡️</div>
            <div style="font-size:18px;font-weight:800;color:#1e293b;">No Reviewed Transactions Yet</div>
            <div style="font-size:13px;color:#64748b;max-width:580px;margin:8px auto 0 auto;line-height:1.6;">
                You have not reviewed any transactions or groups yet.<br>
                Go to the <b>Dashboard</b>, inspect active flagged alerts, and submit an <b>Approve</b> or <b>Reject</b> decision.<br>
                Once reviewed, that transaction or fan-out group will disappear from the Dashboard and appear right here.
            </div>
        </div>
        """)
        st.stop()

    st.write("")

    # ════════════════════════════════════════════════════════════════════
    #  3-COLUMN ARCHITECTURE (MATCHING DASHBOARD LAYOUT)
    #  col_left: Target Cards | col_center: Deep Inspection | col_right: Customer Profile
    # ════════════════════════════════════════════════════════════════════
    col_s2_left, col_s2_center, col_s2_right = st.columns([1.0, 1.8, 1.2])

    with col_s2_left:
        st.html('<div class="section-label">Reviewed Cases Queue</div>')

        valid_ids = [d["target_id"] for d in all_decisions_list]
        if st.session_state.get("s2_selected_id") not in valid_ids:
            st.session_state.s2_selected_id = valid_ids[0]

        for d in all_decisions_list:
            t_id = d.get("target_id", "—")
            acc = d.get("account", "Unknown")
            meta = fraud_data.get_account_meta(acc)
            name = meta.get("entity_name", acc)
            amt = d.get("amount", "—")
            to_info = d.get("to_account", "—")
            dec_val = d.get("decision", "Approve")
            is_sel = (t_id == st.session_state.get("s2_selected_id"))

            if dec_val == "Approve":
                dec_pill = '<span style="background:#dcfce7;color:#15803d;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">✅ APPROVED</span>'
            elif dec_val == "Reject":
                dec_pill = '<span style="background:#fee2e2;color:#b91c1c;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">🚫 REJECTED</span>'
            else:
                dec_pill = '<span style="background:#fef3c7;color:#b45309;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:800;">⚠️ ESCALATED</span>'

            sel_bg = "#eff6ff" if is_sel else "#ffffff"
            sel_border = "#93c5fd" if is_sel else "#e2e8f0"

            st.html(textwrap.dedent(f"""
            <div class="flagged-card" style="background:{sel_bg};border-color:{sel_border};margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="flagged-acc-id">Sender: {acc}</div>
                        <div style="font-size:11px;color:#475569;font-weight:500;margin-top:1px;">{name}</div>
                    </div>
                    {dec_pill}
                </div>
                <div class="flagged-pattern" style="margin-top:6px;">📌 Case {t_id} · Volume: <b>{amt}</b></div>
                <div style="font-size:10.5px;color:#64748b;margin-top:2px;">Counterparty: {to_info}</div>
            </div>
            """))

            if st.button(f"Inspect Case {t_id} →", key=f"s2_btn_rev_{t_id}", use_container_width=True):
                st.session_state.s2_selected_id = t_id
                st.session_state.s2_selected_sub_tx = None
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — Deep Inspection, Routing Flow Table, XAI & Decisions
    # ════════════════════════════════════════════════════════════════════
    with col_s2_center:
        selected_id = st.session_state.get("s2_selected_id")
        if not selected_id:
            st.info("👈 Select a case or transaction from the left panel to inspect predictions, routing flow, and authorizer feedback.")
            st.stop()

        inv = fraud_data.get_investigation_by_id(selected_id)
        tx_match = next((t for t in proc_txs if t["tx_id"] == selected_id), None)

        sender_acc = inv.get("account") or (tx_match.get("from_account") if tx_match else selected_id.replace("GROUP-", ""))
        src_meta = fraud_data.get_account_meta(sender_acc)
        sender_name = src_meta.get("entity_name", sender_acc)
        cur_d = all_decisions.get(selected_id) or all_decisions.get(f"GROUP-{sender_acc}") or get_decision_for_tx(selected_id, sender_acc)

        pred_risk = tx_match.get("risk_tier") if tx_match else inv.get("risk", "High")
        pred_score = tx_match.get("risk_score") if tx_match else inv.get("risk_score", 98)
        if pred_score >= 80:
            pred_risk = "High"
        pred_sig = tx_match.get("gat_prob") if tx_match else inv.get("gat_prob", 0.046942)
        pred_sig_str = format_probability(pred_sig)
        amt_disp = cur_d.get("amount") if cur_d and cur_d.get("amount") not in ["—", "\u2014"] else (tx_match.get("amount_formatted") if tx_match else inv.get("amount_formatted", "$0.00"))
        ts_disp = cur_d.get("timestamp") if cur_d else (tx_match.get("timestamp") if tx_match else inv.get("timestamp", "—"))
        pay_fmt = tx_match.get("payment_format") if tx_match else inv.get("payment_format", "Wire")
        curr_name = tx_match.get("currency") if tx_match else inv.get("payment_currency", "US Dollar")
        uniq_recv = inv.get("unique_receivers") or (tx_match.get("sender_unique_receivers", 1) if tx_match else 1)

        is_fo = (inv.get("unique_receivers", 1) > 1 or (tx_match and tx_match.get("is_fanout")))
        pattern_label = f"⚠️ Max {uniq_recv}-degree Fan-Out" if is_fo else "Direct 1-Hop Transfer"
        target_label_disp = selected_id if selected_id.startswith("GROUP-") else f"TX #{selected_id}"

        st.html(textwrap.dedent(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <div style="font-size:16px;font-weight:800;color:#0f172a;">
                    {target_label_disp}
                    <span style="font-size:12px;font-weight:500;color:#64748b;margin-left:6px;">{pattern_label}</span>
                </div>
                <div style="font-size:12px;color:#475569;margin-top:2px;">
                    Sender: <b>{sender_acc}</b> ({sender_name}) · Bank: <b>{src_meta.get('bank_name', 'Global Bank')}</b> · Timestamp: {ts_disp}
                </div>
            </div>
            <span class="badge-{pred_risk.lower()}">{pred_risk.upper()} RISK · {pred_score}/100</span>
        </div>
        """))

        if st.button("🕸️ View in Graph Network", key="btn_s2_goto_graph", use_container_width=False):
            st.session_state.selected_tx_id = f"GROUP-{sender_acc}"
            st.session_state.goto_graph = True
            st.rerun()

        st.html('<div class="section-label">Transaction Routing Flow (Source & Receivers)</div>')

        fan_rows = fraud_data.get_fan_out_rows(selected_id, include_source=True)
        if not fan_rows:
            r_acc = cur_d.get("to_account") if cur_d else (tx_match.get("to_account", "—") if tx_match else "—")
            r_meta = fraud_data.get_account_meta(r_acc)
            fan_rows = [
                {
                    "sub_tx_id": f"SRC-{sender_acc[:8]}",
                    "role": "Source (Sender)",
                    "account": sender_acc,
                    "to_account": sender_acc,
                    "is_source": True,
                    "amount": amt_disp,
                    "time": ts_disp,
                    "payment_format": pay_fmt,
                    "gat_signal": pred_risk.upper(),
                    "to_entity_name": sender_name,
                    "to_entity_id": src_meta.get("entity_id", f"ENT-{sender_acc[:8]}"),
                    "to_bank_name": src_meta.get("bank_name", "Global Bank"),
                    "to_bank_id": src_meta.get("bank_id", "BNK-001"),
                },
                {
                    "sub_tx_id": selected_id,
                    "role": "Receiver (Hop 1)",
                    "account": r_acc,
                    "to_account": r_acc,
                    "is_source": False,
                    "amount": amt_disp,
                    "time": ts_disp,
                    "payment_format": pay_fmt,
                    "gat_signal": pred_risk.upper(),
                    "to_entity_name": r_meta.get("entity_name", f"Account {r_acc}"),
                    "to_entity_id": r_meta.get("entity_id", f"ENT-{r_acc[:8]}"),
                    "to_bank_name": r_meta.get("bank_name", "Global Bank"),
                    "to_bank_id": r_meta.get("bank_id", "BNK-002"),
                }
            ]

        df_fan = pd.DataFrame(fan_rows)
        df_display = df_fan[["role", "account", "to_entity_name", "amount", "time", "payment_format", "gat_signal"]].copy()
        df_display.columns = ["Role", "Account Number", "Entity Name", "Amount", "Timestamp", "Payment Format", "GAT Signal"]

        s2_event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="s2_fan_table"
        )

        s2_sel_rows = s2_event.selection.get("rows", []) if s2_event.selection else []
        if s2_sel_rows and len(fan_rows) > s2_sel_rows[0]:
            st.session_state.s2_selected_sub_tx = fan_rows[s2_sel_rows[0]]

        st.html("""
        <div style="font-size:10.5px;color:#94a3b8;margin-top:4px;">
            🖱️ Click any row (Source Sender or Hop-1 Receiver) to inspect its behavioral customer profile on the right.
        </div>
        """)

        # XAI Neural Explanation Box
        xai_box_class = "xai-box-high" if pred_risk == "High" else ""
        st.html(textwrap.dedent(f"""
        <div class="xai-box {xai_box_class}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div class="xai-title">🔎 GAT Neural Detection Analysis — {target_label_disp}</div>
                <span class="badge-{pred_risk.lower()}">{pred_risk.upper()}</span>
            </div>
            <ul class="xai-list">
                <li>GAT Graph Attention Network (PyG): {pred_risk.upper()} — Calibrated Risk Score: {pred_score}/100 (Raw Neural Sigmoid: {pred_sig_str})</li>
                <li>Cumulative Transaction Volume: {amt_disp} ({curr_name}) via {pay_fmt}</li>
                <li>Connected Counterparties: {len(fan_rows)-1 if len(fan_rows)>1 else 1} unique receiver(s) in active transaction trail</li>
                {f"<li style='color:#dc2626;font-weight:700;'>🚨 {tx_match.get('retroactive_reason')}</li>" if tx_match and tx_match.get('retroactive_escalated') else ""}
            </ul>
        </div>
        """))

        # Authorizer Decision Panel
        if cur_d:
            dec_val = cur_d.get("decision", "Approve")
            is_app = (dec_val == "Approve")
            is_esc = (dec_val == "Escalate")
            b_color = "#10b981" if is_app else ("#f59e0b" if is_esc else "#ef4444")
            bg_color = "#f0fdf4" if is_app else ("#fffbeb" if is_esc else "#fef2f2")
            txt_color = "#166534" if is_app else ("#b45309" if is_esc else "#991b1b")
            verdict_title = "✅ AUTHORIZER DECISION: APPROVE" if is_app else ("⚠️ AUTHORIZER DECISION: ESCALATE (SAR)" if is_esc else "🚫 AUTHORIZER DECISION: REJECT")

            st.html(textwrap.dedent(f"""
            <div style="background:{bg_color};border:2px solid {b_color};border-radius:10px;padding:16px 20px;margin-top:14px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:15px;font-weight:900;color:{txt_color};">
                        {verdict_title}
                    </span>
                    <span style="font-size:11.5px;color:#64748b;">🕒 {cur_d.get('timestamp', '—')}</span>
                </div>
                <div style="font-size:12.5px;color:#0f172a;margin-top:10px;background:#ffffff;padding:10px 14px;border-radius:6px;border-left:4px solid {b_color};">
                    <b>Authorizer Feedback Notes:</b> {cur_d.get('notes') or '(No compliance notes entered)'}
                </div>
                {f'<div style="font-size:11.5px;color:#2563eb;margin-top:6px;"><b>Revision Remarks:</b> {cur_d.get("revision_remark")}</div>' if cur_d.get("is_revised") and cur_d.get("revision_remark") else ''}
            </div>
            """))

            rev_key = f"s2_rev_active_{selected_id}"
            if not st.session_state.get(rev_key, False):
                if st.button(f"✏️ Revise Decision for {target_label_disp}", key=f"btn_s2_open_rev_{selected_id}", use_container_width=True):
                    st.session_state[rev_key] = True
                    st.rerun()
            else:
                st.html("""
                <div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                    <div style="font-size:13px;font-weight:800;color:#92400e;">✏️ Revise Authorizer Decision & Feedback</div>
                    <div style="font-size:11.5px;color:#78350f;">Update the verdict or compliance notes. Audit trail will be preserved.</div>
                </div>
                """)
                r_c1, r_c2 = st.columns([1, 1.3])
                with r_c1:
                    rev_idx = 0 if is_app else (2 if is_esc else 1)
                    new_choice = st.radio(
                        "Revised Verdict",
                        ["Approve", "Reject", "Escalate"],
                        index=rev_idx,
                        format_func=lambda c: "✅ Approve (Legitimate)" if c == "Approve" else ("🚫 Reject (Laundering / Block)" if c == "Reject" else "⚠️ Escalate (Senior Review / SAR)"),
                        key=f"radio_s2_rev_{selected_id}"
                    )
                with r_c2:
                    new_notes = st.text_area(
                        "Updated Feedback Notes",
                        value=cur_d.get("notes", ""),
                        key=f"area_s2_rev_{selected_id}",
                        height=90
                    )
                rev_remark = st.text_input("Reason for Revision", value="Authorizer compliance re-audit", key=f"inp_s2_remark_{selected_id}")
                b_r1, b_r2 = st.columns([1, 1])
                with b_r1:
                    if st.button("💾 Save Revised Decision", key=f"btn_save_s2_rev_{selected_id}", type="primary", use_container_width=True):
                        fraud_data.revise_auditor_decision(selected_id, new_choice, new_notes, rev_remark)
                        st.session_state[rev_key] = False
                        st.toast(f"Decision for {target_label_disp} updated to {new_choice}!", icon="✅")
                        st.rerun()
                with b_r2:
                    if st.button("Cancel", key=f"btn_cancel_s2_rev_{selected_id}", use_container_width=True):
                        st.session_state[rev_key] = False
                        st.rerun()

        else:
            st.html(textwrap.dedent(f"""
            <div class="human-decision-panel">
                <div class="human-decision-title">
                    🏦 Authorised Bank Authorizer Decision Required — {target_label_disp}
                </div>
                <div style="font-size:12px;color:#92400e;margin-bottom:8px;">
                    Risk Score: <b>{pred_score}/100</b> — Review this transaction and record an authoritative compliance verdict.
                </div>
            </div>
            """))

            d_c1, d_c2 = st.columns([1, 1.2])
            with d_c1:
                choice = st.radio(
                    "**Authorizer Decision**",
                    ["Approve", "Reject", "Escalate"],
                    index=1 if pred_risk in ["High", "Medium"] else 0,
                    format_func=lambda c: "✅ Approve (Legitimate)" if c == "Approve" else ("🚫 Reject (Laundering / Block)" if c == "Reject" else "⚠️ Escalate (Senior Review / SAR)"),
                    key=f"radio_pred_{selected_id}"
                )
            with d_c2:
                notes = st.text_area(
                    "**Authorizer Feedback Notes**",
                    placeholder="State verified counterparty rationale, KYC validation, or structuring/fan-out justification...",
                    key=f"note_pred_{selected_id}",
                    height=120
                )

            if st.button(f"📋 Submit Decision for {target_label_disp}", type="primary", use_container_width=True, key=f"btn_sub_{selected_id}"):
                fraud_data.submit_auditor_decision(selected_id, choice, notes, target_type="group" if selected_id.startswith("GROUP-") else "transaction")
                st.toast(f"Decision saved for {target_label_disp} ({choice})!", icon="✅")
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Customer Behavioral Profiling (Identical to Dashboard)
    # ════════════════════════════════════════════════════════════════════
    with col_s2_right:
        sub_tx = st.session_state.get("s2_selected_sub_tx")
        if sub_tx is None or (fan_rows and sub_tx.get("account") not in [r.get("account") for r in fan_rows]):
            sub_tx = fan_rows[0] if fan_rows else {
                "account": sender_acc,
                "to_account": sender_acc,
                "is_source": True,
                "to_entity_name": sender_name,
                "to_bank_name": src_meta.get("bank_name", "Global Bank"),
                "to_bank_id": src_meta.get("bank_id", "BNK-001"),
                "to_entity_id": src_meta.get("entity_id", f"ENT-{sender_acc[:8]}"),
                "payment_format": pay_fmt,
                "gat_signal": pred_risk.upper(),
                "amount": amt_disp,
            }

        acc_num = sub_tx.get("account") or sub_tx.get("to_account", sender_acc)
        is_src = sub_tx.get("is_source", True)
        profile_label = "Source Sender Profile" if is_src else "Receiver Profile"

        try:
            profile = fraud_data.get_customer_profile(acc_num)
        except Exception:
            profile = {}
        if not isinstance(profile, dict):
            profile = {}

        profile_name = profile.get("name") if profile.get("name") not in ("—", "Unknown", None, "") else sub_tx.get("to_entity_name", acc_num)
        bank_name = profile.get("bank_name") if profile.get("bank_name") not in ("—", None, "") else sub_tx.get("to_bank_name", "Global Bank")
        bank_id = profile.get("bank_id") if profile.get("bank_id") not in ("—", None, "") else sub_tx.get("to_bank_id", "BNK-001")
        entity_id = profile.get("entity_id") if profile.get("entity_id") not in ("—", None, "") else sub_tx.get("to_entity_id", "ENT-001")
        tot_inc = profile.get("total_incoming") if profile.get("total_incoming") not in ("—", None, "") else sub_tx.get("amount", "$0.00")
        tot_out = profile.get("total_outgoing", "—")
        avg_in = profile.get("avg_incoming_amount", "—")
        avg_out = profile.get("avg_outgoing_amount", "—")
        max_in = profile.get("max_incoming_amount", "—")
        max_out = profile.get("max_outgoing_amount", "—")
        in_tx = profile.get("previous_incoming", "—")
        out_tx = profile.get("previous_outgoing", "—")
        uniq_snds = profile.get("unique_senders", "—")
        uniq_recs = profile.get("unique_receivers", "—")
        tot_deg = profile.get("total_degree", "—")
        net_flow = profile.get("net_flow", "—")

        risk_tier = profile.get("risk_tier", f"{pred_risk} Risk")
        tier_color = "#dc2626" if "High" in risk_tier else ("#d97706" if "Medium" in risk_tier else "#16a34a")
        payment_fmt = str(sub_tx.get("payment_format", pay_fmt))

        beh_rows = [
            ("Total Incoming Amount", tot_inc),
            ("Average Incoming Amount", avg_in),
            ("Maximum Incoming Amount", max_in),
            ("Incoming Transactions", str(in_tx)),
            ("Unique Senders", str(uniq_snds)),
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
#  AUTHORIZER DECISIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page in ["Decisions", "Authorizer Decisions"]:
    st.html(textwrap.dedent("""
    <div>
        <h2 style="font-size:22px;font-weight:800;color:#0f172a;margin:0;">🏦 Authorizer Decisions & Audit History</h2>
        <div style="font-size:13px;color:#64748b;margin-top:2px;">
            Full compliance ledger of all decisions rendered by the bank authorizer. Review notes, inspect approval/rejection trails, and revise decisions if needed.
        </div>
    </div>
    """))

    sb_status = fraud_data.get_supabase_status()
    sb_conn = sb_status.get("connected", False)
    sb_color = "#10b981" if sb_conn else "#64748b"
    sb_bg = "#ecfdf5" if sb_conn else "#f8fafc"
    sb_border = "#a7f3d0" if sb_conn else "#e2e8f0"
    sb_icon = "🟢 Supabase Cloud Active" if sb_conn else "⚪ Standby / Local Cache"
    st.html(f"""
    <div style="background:{sb_bg};border:1px solid {sb_border};border-radius:8px;padding:8px 14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:12px;color:#1e293b;">
            <b>Database Engine:</b> {sb_status.get('provider')} &nbsp;|&nbsp; <b>Storage Source:</b> <code>{sb_status.get('decisions_source')}</code>
        </div>
        <div style="font-size:11px;font-weight:700;color:{sb_color};background:#ffffff;padding:3px 8px;border-radius:4px;border:1px solid {sb_border};">
            {sb_icon}
        </div>
    </div>
    """)

    all_decs = fraud_data.get_all_auditor_decisions()
    if not all_decs:
        st.info("No authorizer decisions have been submitted yet. Review predictions on Screen 2 or the main Dashboard to submit decisions.")
    else:
        tot_d = len(all_decs)
        app_d = sum(1 for d in all_decs if "Approve" in str(d.get("decision", "")))
        rej_d = sum(1 for d in all_decs if "Reject" in str(d.get("decision", "")))
        esc_d = sum(1 for d in all_decs if "Escalate" in str(d.get("decision", "")))
        rev_d = sum(1 for d in all_decs if d.get("is_revised", False))

        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        with col_k1:
            st.metric("Total Decisions Rendered", f"{tot_d}")
        with col_k2:
            st.metric("Approved (Label 0)", f"{app_d}")
        with col_k3:
            st.metric("Rejected (Label 1)", f"{rej_d}")
        with col_k4:
            st.metric("Escalated (Label 1)", f"{esc_d}")
        with col_k5:
            st.metric("Revised Decisions", f"{rev_d}")

        filter_choice = st.pills("Filter Decisions", ["All", "Approved", "Rejected", "Escalated", "Revised"], default="All")

        filtered = all_decs
        if filter_choice == "Approved":
            filtered = [d for d in all_decs if "Approve" in str(d.get("decision", ""))]
        elif filter_choice == "Rejected":
            filtered = [d for d in all_decs if "Reject" in str(d.get("decision", ""))]
        elif filter_choice == "Escalated":
            filtered = [d for d in all_decs if "Escalate" in str(d.get("decision", ""))]
        elif filter_choice == "Revised":
            filtered = [d for d in all_decs if d.get("is_revised", False)]

        # Inline Revision Card if a target is chosen for revision
        if st.session_state.revision_target_id:
            rev_target = st.session_state.revision_target_id
            target_data = next((d for d in all_decs if d["target_id"] == rev_target), None)
            if target_data:
                st.markdown("---")
                st.html(f"""
                <div style="background:#f0f9ff;border:2px solid #38bdf8;border-radius:10px;padding:16px;margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:15px;font-weight:800;color:#0369a1;">✏️ Revise Authorizer Decision: {rev_target}</div>
                        <span style="font-size:11px;color:#0284c7;font-weight:600;">Current Decision: {target_data.get('decision')} (Label {target_data.get('training_label', 0)})</span>
                    </div>
                    <div style="font-size:12px;color:#334155;margin-top:4px;">
                        Adjust the verdict, update compliance notes, and provide revision justification. The training label will update accordingly.
                    </div>
                </div>
                """)

                c_rev1, c_rev2 = st.columns([1, 1])
                with c_rev1:
                    cur_d_str = str(target_data.get("decision", ""))
                    idx_choice = 0 if "Approve" in cur_d_str else (2 if "Escalate" in cur_d_str else 1)
                    new_choice = st.radio(
                        "New Authorizer Verdict",
                        ["Approve", "Reject", "Escalate"],
                        index=idx_choice,
                        format_func=lambda c: "✅ Approve (Legitimate)" if c == "Approve" else ("🚫 Reject (Laundering / Block)" if c == "Reject" else "⚠️ Escalate (Senior Review / SAR)"),
                        key="radio_rev_choice"
                    )
                    rev_remark = st.text_input(
                        "Revision Justification Remarks",
                        placeholder="e.g., Mistakenly approved earlier; secondary review of fan-out cluster confirmed laundering.",
                        key="input_rev_remark"
                    )
                with c_rev2:
                    new_notes = st.text_area(
                        "Updated Authorizer Notes",
                        value=target_data.get("notes", ""),
                        key="area_rev_notes",
                        height=120
                    )

                btn_c1, btn_c2 = st.columns([1, 1])
                with btn_c1:
                    if st.button("💾 Save Revised Decision", type="primary", use_container_width=True):
                        fraud_data.revise_auditor_decision(rev_target, new_choice, new_notes, rev_remark)
                        st.session_state.revision_target_id = None
                        st.toast(f"Decision for {rev_target} successfully revised!", icon="🔄")
                        st.rerun()
                with btn_c2:
                    if st.button("Cancel Revision", use_container_width=True):
                        st.session_state.revision_target_id = None
                        st.rerun()
                st.markdown("---")

        # Decisions Card List
        for d in filtered:
            t_id = d.get("target_id", "—")
            dec_str = d.get("decision", "—")
            is_app = "Approve" in dec_str
            is_esc = "Escalate" in dec_str
            t_label = d.get("training_label", 0 if is_app else 1)
            b_class = "badge-approved" if is_app else ("badge-medium" if is_esc else "badge-blocked")
            is_rev = d.get("is_revised", False)
            rev_badge = '<span class="badge-revised">REVISED DECISION</span>' if is_rev else ''

            with st.container():
                st.html(f"""
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:14px;font-weight:800;color:#0f172a;">{t_id}</span>
                            <span style="font-size:11.5px;color:#475569;margin-left:8px;">GAT Probability: <b>{format_probability(d.get('gat_prob', 0.046942))}</b> · Risk Level: <b>{d.get('risk_level', 'High')}</b> · Amount: <b>{d.get('amount', '—')}</b></span>
                        </div>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <span style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;background:#f1f5f9;color:#334155;border:1px solid #cbd5e1;">Training Label: {t_label}</span>
                            {rev_badge}
                            <span class="{b_class}">HUMAN DECISION: {dec_str.upper()}</span>
                        </div>
                    </div>
                    <div style="font-size:12px;color:#334155;margin-top:6px;background:#f8fafc;padding:8px 12px;border-radius:6px;border-left:3px solid #cbd5e1;">
                        <b>Remarks:</b> {d.get('remarks') or d.get('notes') or '(No remarks provided)'}
                    </div>
                    {f'<div style="font-size:11.5px;color:#2563eb;margin-top:4px;"><b>Revision Remarks:</b> {d.get("revision_remark")}</div>' if is_rev and d.get("revision_remark") else ''}
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                        <span style="font-size:11px;color:#94a3b8;">🕒 Decision Timestamp: {d.get('timestamp', '—')}</span>
                    </div>
                </div>
                """)
                if st.button(f"✏️ Revise Decision for {t_id}", key=f"btn_rev_{t_id}"):
                    st.session_state.revision_target_id = t_id
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  OVERNIGHT RETRAINING PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Overnight Retraining":
    st.html(textwrap.dedent("""
    <div>
        <h2 style="font-size:22px;font-weight:800;color:#0f172a;margin:0;">🌙 Overnight Model Retraining Pipeline</h2>
        <div style="font-size:13px;color:#64748b;margin-top:2px;">
            Human-in-the-Loop Continuous Learning: The Graph Attention Network (GAT) incorporates human bank authorizer decisions through a scheduled overnight fine-tuning pipeline.
        </div>
    </div>
    <br>
    """))

    st.html("""
    <div style="background:#eff6ff;border:1.5px solid #93c5fd;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:13px;font-weight:700;color:#1e40af;">📌 Mentor Architecture Overview: Overnight Retraining</div>
        <div style="font-size:12px;color:#1e3a8a;margin-top:4px;line-height:1.6;">
            1. Predictions and risk scores are generated on incoming transactions.<br>
            2. Authorised bank compliance officers review flagged transactions and record authoritative ground-truth verdicts.<br>
            3. When at least 50 human authorizer decisions are accumulated, GitHub Actions triggers overnight retraining at 02:00 AM IST.<br>
            4. The fine-tuned weights are saved to <code>backend/GAT/gat_aml_retrained.pt</code> and automatically synchronized to <b>Hugging Face Hub</b>.
        </div>
    </div>
    """)

    sched_info = fraud_data.get_scheduler_status()
    st.html(f"""
    <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;padding:12px 18px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:12px;font-weight:700;color:#0f172a;">⚡ GitHub Actions Overnight Retraining Pipeline</div>
            <div style="font-size:11.5px;color:#64748b;margin-top:2px;">
                Engine: <b>{sched_info.get('engine')}</b> &nbsp;|&nbsp; Schedule: <b>{sched_info.get('schedule')}</b> &nbsp;|&nbsp; Trigger: <b style="color:#0284c7;">{sched_info.get('threshold')}</b> &nbsp;|&nbsp; Sync: <b>{sched_info.get('sync_target')}</b>
            </div>
        </div>
        <div style="font-size:11px;background:#e0f2fe;color:#0369a1;padding:4px 10px;border-radius:6px;font-weight:600;">
            CRON: 02:00 AM IST
        </div>
    </div>
    """)

    all_decs = fraud_data.get_all_auditor_decisions()
    tot_samples = len(all_decs)
    app_cnt = sum(1 for d in all_decs if "Approve" in str(d.get("decision", "")))
    fraud_cnt = sum(1 for d in all_decs if "Block" in str(d.get("decision", "")) or "Laundering" in str(d.get("decision", "")))

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Audited Training Samples", f"{tot_samples}")
    with col_s2:
        st.metric("Legitimate Samples (y=0)", f"{app_cnt}")
    with col_s3:
        st.metric("Laundering Samples (y=1)", f"{fraud_cnt}")
    with col_s4:
        retrained_exists = os.path.exists(fraud_data.RETRAINED_GAT_PATH)
        st.metric("Model Checkpoint", "Retrained Active" if retrained_exists else "Base Checkpoint")

    st.write("")
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        epochs_sel = st.slider("Overnight Retraining Epochs", min_value=3, max_value=15, value=8)
    with col_r2:
        lr_sel = st.selectbox("Fine-Tuning Learning Rate", [0.0005, 0.001, 0.002], index=1)

    if st.button("🌙 Run Overnight Retraining Batch", type="primary", use_container_width=True):
        with st.spinner("Executing overnight batch fine-tuning on PyG Graph Attention Network..."):
            report = fraud_data.run_overnight_retraining(epochs=epochs_sel, lr=lr_sel)
            st.session_state.retrain_result = report
            if report.get("status") == "success":
                st.toast("🎉 Overnight Retraining completed successfully!", icon="✅")
            else:
                st.toast(report.get("message", "Error in retraining"), icon="⚠️")
            st.rerun()

    report = st.session_state.get("retrain_result")
    if not report:
        past_hist = fraud_data.get_retraining_history()
        if past_hist:
            report = past_hist[-1]

    if report and report.get("status") == "success":
        st.markdown("---")
        st.markdown("### 📊 Retraining Results & Metric Comparison")

        pre_l = report["pre_loss"]
        post_l = report["post_loss"]
        pre_a = report["pre_accuracy"]
        post_a = report["post_accuracy"]
        pre_p = report["pre_precision"]
        post_p = report["post_precision"]
        pre_r = report["pre_recall"]
        post_r = report["post_recall"]

        delta_l = round(post_l - pre_l, 4)
        delta_a = round(post_a - pre_a, 1)
        delta_p = round(post_p - pre_p, 1)
        delta_r = round(post_r - pre_r, 1)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("BCE Training Loss", f"{post_l:.4f}", f"{delta_l:.4f}", delta_color="inverse")
        with m2:
            st.metric("Model Accuracy", f"{post_a:.1f}%", f"{delta_a:+.1f}%")
        with m3:
            st.metric("Model Precision", f"{post_p:.1f}%", f"{delta_p:+.1f}%")
        with m4:
            st.metric("Model Recall", f"{post_r:.1f}%", f"{delta_r:+.1f}%")

        loss_hist = report.get("loss_history", [])
        if loss_hist:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(loss_hist) + 1)),
                y=loss_hist,
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8, color="#1e40af"),
                name="BCE Loss"
            ))
            fig_loss.update_layout(
                title="<b>Overnight Fine-Tuning Convergence (Loss vs. Epoch)</b>",
                xaxis_title="Training Epoch",
                yaxis_title="BCE With Logits Loss",
                template="plotly_white",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("#### Before vs. After Retraining Metrics")
        df_comp = pd.DataFrame([
            {"Metric": "BCE Loss", "Before Retraining (Base GAT)": f"{pre_l:.4f}", "After Overnight Retraining": f"{post_l:.4f}", "Improvement": f"{delta_l:.4f} (Loss Reduced)"},
            {"Metric": "Accuracy", "Before Retraining (Base GAT)": f"{pre_a:.1f}%", "After Overnight Retraining": f"{post_a:.1f}%", "Improvement": f"{delta_a:+.1f}%"},
            {"Metric": "Precision", "Before Retraining (Base GAT)": f"{pre_p:.1f}%", "After Overnight Retraining": f"{post_p:.1f}%", "Improvement": f"{delta_p:+.1f}%"},
            {"Metric": "Recall", "Before Retraining (Base GAT)": f"{pre_r:.1f}%", "After Overnight Retraining": f"{post_r:.1f}%", "Improvement": f"{delta_r:+.1f}%"},
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.html(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px 16px;margin-top:12px;">
            <div style="font-size:12.5px;font-weight:700;color:#166534;">
                ✅ GAT Model Successfully Retrained and Deployed
            </div>
            <div style="font-size:11.5px;color:#14532d;margin-top:2px;">
                Trained on <b>{report.get('total_samples')} human-verified cases</b> ({report.get('approved_count')} approved, {report.get('fraud_count')} laundering). Checkpoint saved at <code>{report.get('model_path')}</code>.
            </div>
        </div>
        """)

    # GAT Model Version Registry Section
    st.markdown("---")
    st.markdown("### 🏷️ GAT Model Version Registry & Artifact Lineage")
    reg_data = fraud_data.get_model_registry()
    if reg_data and "models" in reg_data:
        reg_rows = []
        for m in reg_data.get("models", []):
            mets = m.get("metrics") or {}
            perf_summary = f"Loss: {mets.get('training_loss_final', '—')} | Acc: {mets.get('accuracy', '—')}%" if mets else "Base Pretrained (10M Tx)"
            reg_rows.append({
                "Version": m.get("version"),
                "Model Type": m.get("model_type"),
                "Weights File": m.get("weights_file"),
                "Status": m.get("status"),
                "Training Source": m.get("training_source"),
                "Performance": perf_summary,
                "Registered At": m.get("created_at"),
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NETWORK PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Graph Network":
    st.subheader(" Money Trail")
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

        sel_tx = st.selectbox(
            "Select Fan-out Group Network to Inspect",
            group_options,
            index=default_idx
        )

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
                <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('model_used', 'PyTorch Geometric GAT AML Model')}</div>
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
        tx_risk = tx_info.get("risk", "High")
        tx_score = tx_info.get("risk_score", 92)
        tx_clean_exps = []
        for e in tx_info.get("explanations", []):
            if "payment format" in e.lower() or "currency:" in e.lower():
                continue
            if "risk probability" in e.lower() or "gat graph" in e.lower() or "neural network" in e.lower() or "ensemble" in e.lower():
                tx_clean_exps.append(f"AI Ensemble (GAT Graph Attention + LightGBM Motif): {tx_risk.upper()} ({tx_score}% Confidence · Risk Score: {tx_score}/100)")
            else:
                tx_clean_exps.append(e)

        st.html(textwrap.dedent(f"""
        <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:14px 16px;">
            <div style="font-weight:700;font-size:13px;color:#dc2626;margin-bottom:8px;">⚠️ Why was this flagged?</div>
            <ul style="margin:0;padding-left:18px;font-size:12px;color:#334155;line-height:1.8;">
                {"".join([f"<li>{e}</li>" for e in tx_clean_exps])}
            </ul>
        </div>
        """))

# ─── Automated Streaming Loop ─────────────────────────────────────────────
if st.session_state.get("is_streaming", False):
    status_now = fraud_data.get_stream_status()
    if status_now["current_idx"] < status_now["total_txs"]:
        time.sleep(st.session_state.get("stream_delay", 1.0))
        fraud_data.step_stream(1)
        st.rerun()
    else:
        st.session_state.is_streaming = False
        st.toast(f"🎉 Streaming Completed! All {status_now['total_txs']} transactions processed.", icon="✅")
        st.rerun()




