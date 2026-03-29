"""
Interactive Analyst Dashboard Demo

Run with: streamlit run src/demo/analyst_dashboard_streamlit.py

Features:
- Live alert queue with filters
- Alert detail view with customer info
- Model predictions visualization
- Human-in-the-loop decision making
- Victim support protocol
- Performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Load reason code descriptions from config
_config_path = Path(__file__).resolve().parents[2] / 'configs' / 'japan_calibration.yaml'
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_reason_descriptions = _config.get('reason_codes', {})

# Page config
st.set_page_config(
    page_title="Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'alerts' not in st.session_state:
    # Generate demo alerts
    st.session_state.alerts = pd.DataFrame({
        'alert_id': ['FRD-001', 'FRD-002', 'FRD-003', 'FRD-004', 'FRD-005'],
        'priority': ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'LOW'],
        'customer_id': ['CUST-ELDERLY-001', 'CUST-BIZ-045', 'CUST-REG-123', 'CUST-REG-456', 'CUST-NEW-789'],
        'customer_age': [72, 45, 35, 28, 42],
        'vulnerable': ['Elderly', 'None', 'None', 'Low Income', 'None'],
        'amount': [5000000, 1200000, 800000, 500000, 300000],
        'recipient': ['ACC-SCAMMER-999', 'ACC-SUPPLIER-123', 'ACC-FRIEND-456', 'ACC-SHOP-789', 'ACC-UTIL-012'],
        'ensemble_score': [0.92, 0.68, 0.55, 0.48, 0.35],
        'balance_drain': [0.85, 0.45, 0.30, 0.60, 0.15],
        'time_in_queue': [3, 15, 45, 120, 180],
        'status': ['Pending', 'Pending', 'Pending', 'Pending', 'Pending'],
        'reason_codes': [
            'AR02, TR03, CM02',
            'TR01, TR04',
            'TR01',
            'TR03',
            'None'
        ]
    })

if 'reviewed_count' not in st.session_state:
    st.session_state.reviewed_count = 89

if 'analyst_id' not in st.session_state:
    st.session_state.analyst_id = 'analyst_001'

# Sidebar
with st.sidebar:
    st.title("🛡️ Fraud Detection System")

    st.markdown("---")

    st.markdown(f"**Analyst:** {st.session_state.analyst_id}")

    st.metric("Alerts Reviewed Today", st.session_state.reviewed_count, "+12 vs yesterday")
    st.metric("Avg Review Time", "18 min", "-2 min (GOOD ✅)")
    st.metric("Accuracy Rate", "96.2%", "+0.5%")

    st.markdown("---")

    st.subheader("Queue Summary")
    critical_count = len(st.session_state.alerts[st.session_state.alerts['priority'] == 'CRITICAL'])
    high_count = len(st.session_state.alerts[st.session_state.alerts['priority'] == 'HIGH'])
    medium_count = len(st.session_state.alerts[st.session_state.alerts['priority'] == 'MEDIUM'])
    low_count = len(st.session_state.alerts[st.session_state.alerts['priority'] == 'LOW'])

    st.write(f"🔴 **CRITICAL:** {critical_count} (SLA: 30 min)")
    st.write(f"🟠 **HIGH:** {high_count} (SLA: 2 hours)")
    st.write(f"🟡 **MEDIUM:** {medium_count} (SLA: 24 hours)")
    st.write(f"⚪ **LOW:** {low_count} (SLA: 48 hours)")

    st.markdown("---")

    if st.button("🔄 Refresh Queue"):
        st.rerun()

# Main panel
st.title("Fraud Alert Review Queue")

# Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    priority_filter = st.selectbox("Priority", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

with col2:
    vulnerable_filter = st.selectbox("Vulnerable Pop", ["All", "Elderly", "Disabled", "Low Income", "None"])

with col3:
    status_filter = st.selectbox("Status", ["Pending", "In Review", "Completed", "All"])

with col4:
    sort_by = st.selectbox("Sort By", ["Time (Oldest First)", "Priority", "Amount"])

# Filter alerts
filtered_alerts = st.session_state.alerts.copy()

if priority_filter != "All":
    filtered_alerts = filtered_alerts[filtered_alerts['priority'] == priority_filter]

if vulnerable_filter != "All":
    filtered_alerts = filtered_alerts[filtered_alerts['vulnerable'] == vulnerable_filter]

if status_filter != "All":
    filtered_alerts = filtered_alerts[filtered_alerts['status'] == status_filter]

# Display alert queue
st.markdown("---")
st.subheader(f"📋 {len(filtered_alerts)} Alerts in Queue")

for idx, alert in filtered_alerts.iterrows():
    priority_emoji = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '⚪'
    }[alert['priority']]

    vulnerable_badge = f"🧓 {alert['vulnerable']}" if alert['vulnerable'] != 'None' else ""

    with st.expander(
        f"{priority_emoji} **{alert['priority']}** - Alert {alert['alert_id']} - ¥{alert['amount']:,} - {alert['time_in_queue']} min {vulnerable_badge}",
        expanded=(idx == 0)  # Expand first alert
    ):
        # Alert detail view
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Customer Information")
            st.write(f"**Customer ID:** {alert['customer_id']}")
            st.write(f"**Age:** {alert['customer_age']}")
            if alert['vulnerable'] != 'None':
                st.error(f"**🚨 VULNERABLE POPULATION:** {alert['vulnerable']}")

        with col2:
            st.markdown("### 💰 Transaction Details")
            st.write(f"**Amount:** ¥{alert['amount']:,}")
            st.write(f"**Recipient:** {alert['recipient']}")
            st.write(f"**Balance Drain:** {alert['balance_drain']*100:.0f}%")
            st.write(f"**Time in Queue:** {alert['time_in_queue']} minutes")

        st.markdown("---")

        # Model predictions
        st.markdown("### 🤖 Ensemble Model Predictions")

        # Generate individual model scores (simulate)
        balance_score = min(alert['ensemble_score'] + np.random.uniform(-0.05, 0.05), 1.0)
        velocity_score = min(alert['ensemble_score'] + np.random.uniform(-0.1, 0.05), 1.0)
        temporal_score = min(alert['ensemble_score'] + np.random.uniform(-0.05, 0.08), 1.0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Balance Model", f"{balance_score:.3f}")
            st.progress(balance_score)

        with col2:
            st.metric("Velocity Model", f"{velocity_score:.3f}")
            st.progress(velocity_score)

        with col3:
            st.metric("Temporal Model", f"{temporal_score:.3f}")
            st.progress(temporal_score)

        with col4:
            st.metric("**Ensemble Score**", f"{alert['ensemble_score']:.3f}", delta="CRITICAL" if alert['ensemble_score'] >= 0.8 else "HIGH" if alert['ensemble_score'] >= 0.6 else "MEDIUM")
            st.progress(alert['ensemble_score'])

        st.markdown("---")

        # Risk factors
        st.markdown("### 🚨 Risk Factors (Reason Codes)")
        reason_codes = alert['reason_codes'].split(', ')
        for code in reason_codes:
            desc = _reason_descriptions.get(code)
            if desc:
                st.write(f"• **{code}:** {desc}")

        st.markdown("---")

        # Analyst decision
        st.markdown("### 📝 Analyst Decision")

        decision_key = f"decision_{alert['alert_id']}"
        notes_key = f"notes_{alert['alert_id']}"

        col1, col2 = st.columns([2, 3])

        with col1:
            decision = st.radio(
                "Decision",
                ["APPROVE", "BLOCK", "REQUEST MORE INFO", "ESCALATE TO SENIOR"],
                key=decision_key
            )

        with col2:
            notes = st.text_area(
                "Analyst Notes (Required)",
                placeholder="Document your decision and reasoning...",
                height=100,
                key=notes_key
            )

        # Show victim support if blocking vulnerable customer
        if decision == "BLOCK" and alert['vulnerable'] != 'None':
            st.warning("### 🆘 Victim Support Protocol Will Be Initiated")
            st.write("**Automatic Actions:**")
            st.write("✅ Transaction BLOCKED")
            st.write("✅ Partial freeze activated (outgoing transfers only)")
            st.write("✅ Customer notification sent (gentle language)")
            st.write("✅ Victim resources email queued")

            st.write("**Required Follow-up (You must complete):**")
            st.write("1. ☎️ Phone call within 30 minutes")
            st.write("2. 📄 Send victim resources (police hotline, legal aid)")
            st.write("3. 🔍 Fast-track investigation (4 hour target)")

        # Submit button
        if st.button(f"✅ Submit Decision for {alert['alert_id']}", key=f"submit_{alert['alert_id']}"):
            if not notes:
                st.error("❌ Please add notes documenting your decision")
            else:
                # Update alert status
                st.session_state.alerts.loc[idx, 'status'] = 'Completed'
                st.session_state.reviewed_count += 1

                st.success(f"✅ Decision submitted: **{decision}**")
                st.info(f"📝 Notes: {notes}")

                if decision == "BLOCK" and alert['vulnerable'] != 'None':
                    st.warning("🆘 Victim support protocol has been initiated. Please complete required follow-up actions.")

                # Auto-refresh after 2 seconds
                import time
                time.sleep(2)
                st.rerun()

# Performance metrics at bottom
st.markdown("---")
st.subheader("📊 Performance Metrics (Today)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "45,234", "+2.3%")

with col2:
    st.metric("Fraud Alerts", "127", "-5")

with col3:
    st.metric("False Positive Rate", "4.2%", "-0.3% (GOOD ✅)")

with col4:
    st.metric("Avg Review Time", "18 min", "-2 min")

# Alert distribution chart
st.markdown("---")
st.subheader("📈 Alert Distribution by Priority")

priority_counts = st.session_state.alerts['priority'].value_counts()

fig = px.pie(
    values=priority_counts.values,
    names=priority_counts.index,
    color=priority_counts.index,
    color_discrete_map={
        'CRITICAL': '#ff4444',
        'HIGH': '#ff9944',
        'MEDIUM': '#ffee44',
        'LOW': '#cccccc'
    }
)

st.plotly_chart(fig, use_container_width=True)

# Instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Review alerts in priority order
    2. Expand alert to see details
    3. Review model predictions
    4. Make decision (APPROVE/BLOCK/etc.)
    5. Add detailed notes
    6. Submit decision

    **For vulnerable populations:**
    - Use extra caution
    - Follow victim support protocol
    - Use gentle language
    """)
