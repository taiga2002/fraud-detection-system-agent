"""
Real-Time Monitoring Dashboard

Run with: streamlit run src/demo/monitoring_dashboard_streamlit.py

Features:
- System health metrics
- Fairness monitoring
- Drift detection
- Security alerts
- Performance tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="📊",
    layout="wide"
)

# Generate demo data
@st.cache_data
def generate_demo_data():
    # Last 7 days of fraud alerts
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    fraud_data = pd.DataFrame({
        'date': dates,
        'total_alerts': np.random.randint(100, 150, 7),
        'confirmed_fraud': np.random.randint(40, 70, 7),
        'false_positives': np.random.randint(30, 50, 7)
    })

    # Demographics data
    demographics = pd.DataFrame({
        'age_group': ['18-35', '36-50', '51-65', '66+'],
        'flagged_rate': [0.038, 0.041, 0.043, 0.045],
        'population': [12500, 15000, 10000, 4500]
    })

    # Feature drift data
    features = ['balance_drain_ratio', 'amount_jpy', 'account_age_days', 'txn_count_24h', 'velocity_ratio']
    drift_data = pd.DataFrame({
        'feature': features,
        'psi_score': [0.08, 0.12, 0.18, 0.05, 0.09],
        'status': ['✅ No drift', '✅ Slight change', '🟡 Monitor', '✅ No drift', '✅ No drift']
    })

    # Security incidents
    security_incidents = pd.DataFrame({
        'time': ['14:23', '11:45', '09:12'],
        'type': ['Honeypot Triggered', 'Rate Limit Exceeded', 'Invalid API Key'],
        'user': ['API_KEY_xyz123', 'API_KEY_abc789', 'UNKNOWN'],
        'action': ['Blocked 24h', 'Blocked 1h', 'Rejected'],
        'severity': ['🔴 High', '🟡 Medium', '🟢 Low']
    })

    return fraud_data, demographics, drift_data, security_incidents

fraud_data, demographics, drift_data, security_incidents = generate_demo_data()

# Header
st.title("🛡️ Fraud Detection System - Real-Time Monitoring")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Status:** 🟢 HEALTHY")

# Auto-refresh
if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 System Health", "⚖️ Fairness Monitoring", "📈 Drift Detection", "🔒 Security"])

# TAB 1: System Health
with tab1:
    st.header("System Health Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Transactions Today",
            "45,234",
            delta="+2.3% vs yesterday",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Fraud Alerts",
            "127",
            delta="-5 vs yesterday",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "False Positive Rate",
            "4.2%",
            delta="-0.3% (GOOD ✅)",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Avg Review Time",
            "18 min",
            delta="✅ Under SLA (30 min)",
            delta_color="off"
        )

    st.markdown("---")

    # Alert queue status
    st.subheader("Alert Queue Status")

    queue_data = pd.DataFrame({
        'Priority': ['🔴 CRITICAL', '🟠 HIGH', '🟡 MEDIUM', '⚪ LOW'],
        'Pending': [2, 8, 15, 42],
        'SLA': ['30 min', '2 hours', '24 hours', '48 hours'],
        'Oldest': ['12 min ✅', '45 min ✅', '8 hours ✅', '18 hours ✅']
    })

    st.dataframe(queue_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Fraud alerts over time
    st.subheader("Fraud Alerts Over Time (Last 7 Days)")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fraud_data['date'],
        y=fraud_data['total_alerts'],
        name='Total Alerts',
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        x=fraud_data['date'],
        y=fraud_data['confirmed_fraud'],
        name='Confirmed Fraud',
        marker_color='red'
    ))

    fig.update_layout(
        barmode='overlay',
        xaxis_title='Date',
        yaxis_title='Count',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics table
    st.subheader("Model Performance Metrics")

    perf_data = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Balance Model': [0.945, 0.892, 0.918, 0.976],
        'Velocity Model': [0.932, 0.885, 0.908, 0.968],
        'Temporal Model': [0.928, 0.879, 0.903, 0.972],
        'Ensemble': [0.951, 0.898, 0.924, 0.982]
    })

    st.dataframe(perf_data, use_container_width=True, hide_index=True)

# TAB 2: Fairness Monitoring
with tab2:
    st.header("Fairness & Bias Monitoring")

    st.info("**Last Bias Audit:** 2026-03-19 (Weekly) | **Next:** 2026-03-26 | **Status:** ✅ ALL TESTS PASSED")

    st.markdown("---")

    # Disparate Impact Testing
    st.subheader("Disparate Impact Testing (80% Rule)")

    impact_data = pd.DataFrame({
        'Protected Attribute': ['Age Group', 'Income Level', 'Customer Tenure', 'Region', 'Digital Literacy'],
        'Adverse Impact Ratio': [0.87, 0.92, 0.81, 0.94, 0.88],
        'Status': ['✅ PASS', '✅ PASS', '✅ PASS', '✅ PASS', '✅ PASS']
    })

    # Horizontal bar chart
    fig = go.Figure(go.Bar(
        x=impact_data['Adverse Impact Ratio'],
        y=impact_data['Protected Attribute'],
        orientation='h',
        marker=dict(
            color=['green' if r >= 0.80 else 'red' for r in impact_data['Adverse Impact Ratio']],
            line=dict(color='black', width=1)
        ),
        text=impact_data['Adverse Impact Ratio'].round(2),
        textposition='auto'
    ))

    fig.add_vline(x=0.80, line_dash="dash", line_color="orange", annotation_text="80% Threshold")

    fig.update_layout(
        xaxis_title='Adverse Impact Ratio',
        yaxis_title='',
        height=300,
        xaxis=dict(range=[0, 1.0])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # False Positive Rate by Demographics
    st.subheader("False Positive Rate by Age Group")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(
            demographics,
            x='age_group',
            y='flagged_rate',
            title='FPR by Age Group',
            labels={'flagged_rate': 'False Positive Rate', 'age_group': 'Age Group'},
            color='flagged_rate',
            color_continuous_scale='Reds'
        )

        fig.add_hline(y=demographics['flagged_rate'].mean(), line_dash="dash", line_color="blue", annotation_text="Average")

        fig.update_layout(height=300)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Max Disparity", "0.7%", delta="ACCEPTABLE ✅", delta_color="off")
        st.write("**Analysis:**")
        st.write("• Elderly (66+): 4.5%")
        st.write("• Youngest (18-35): 3.8%")
        st.write("• Difference: 0.7%")
        st.write("")
        st.success("✅ Disparity is within acceptable range (<1%)")

    st.markdown("---")

    # Vulnerable population alerts
    st.subheader("Vulnerable Population Alerts (Today)")

    vuln_data = pd.DataFrame({
        'Group': ['🧓 Elderly (65+)', '♿ Disabled', '💰 Low Income'],
        'Total Alerts': [12, 2, 8],
        'Approved': [9, 1, 6],
        'Blocked': [3, 0, 0],
        'Pending': [0, 1, 2]
    })

    st.dataframe(vuln_data, use_container_width=True, hide_index=True)

    st.info("✅ All vulnerable population cases are under enhanced review with victim support protocols")

# TAB 3: Drift Detection
with tab3:
    st.header("Model Drift Detection")

    st.info("**Last Full Check:** 2026-03-20 06:00 | **Next:** 2026-03-21 06:00 | **Status:** ✅ NO SIGNIFICANT DRIFT")

    st.markdown("---")

    # Feature drift (PSI scores)
    st.subheader("Feature Drift (Population Stability Index)")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()

        colors = ['green' if psi < 0.25 else 'yellow' if psi < 0.35 else 'red' for psi in drift_data['psi_score']]

        fig.add_trace(go.Bar(
            y=drift_data['feature'],
            x=drift_data['psi_score'],
            orientation='h',
            marker=dict(color=colors),
            text=drift_data['psi_score'].round(2),
            textposition='auto'
        ))

        fig.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Retraining Threshold")

        fig.update_layout(
            xaxis_title='PSI Score',
            yaxis_title='Feature',
            height=300,
            xaxis=dict(range=[0, 0.5])
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**PSI Interpretation:**")
        st.write("• < 0.10: No drift ✅")
        st.write("• 0.10-0.25: Slight change 🟡")
        st.write("• ≥ 0.25: Significant drift 🔴")
        st.write("")
        st.write("**Current Status:**")
        for _, row in drift_data.iterrows():
            st.write(f"• {row['feature']}: {row['status']}")

    st.markdown("---")

    # Prediction distribution drift
    st.subheader("Prediction Distribution Drift")

    # Generate synthetic distribution data
    x = np.linspace(0, 1, 50)
    training_dist = np.random.beta(2, 5, 1000)
    production_dist = np.random.beta(2.1, 5, 1000)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=training_dist,
        name='Training',
        opacity=0.7,
        marker_color='blue',
        nbinsx=20
    ))

    fig.add_trace(go.Histogram(
        x=production_dist,
        name='Production',
        opacity=0.7,
        marker_color='orange',
        nbinsx=20
    ))

    fig.update_layout(
        barmode='overlay',
        xaxis_title='Prediction Score',
        yaxis_title='Frequency',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Training and production distributions are well-aligned. No significant drift detected.")

    st.markdown("---")

    # Retraining recommendation
    st.subheader("Retraining Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Features with Drift", "1", delta="Monitor closely 🟡")

    with col2:
        st.metric("Days Since Training", "14", delta="Normal ✅")

    with col3:
        st.metric("Next Review", "7 days", delta="Scheduled ✅")

    st.info("✅ **Recommendation:** Retraining NOT required. Continue monitoring. Next comprehensive review in 7 days.")

# TAB 4: Security
with tab4:
    st.header("Security & Adversarial Defense")

    # API rate limiting stats
    st.subheader("API Rate Limiting (Last 24h)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Requests", "156,234")

    with col2:
        st.metric("Blocked (Rate Limit)", "45", delta="0.03% of total")

    with col3:
        st.metric("Unique Users", "1,234")

    st.markdown("---")

    # Security incidents
    st.subheader("🚨 Security Incidents (Today)")

    # Add severity colors
    def highlight_severity(row):
        if '🔴' in row['severity']:
            return ['background-color: #ffcccc'] * len(row)
        elif '🟡' in row['severity']:
            return ['background-color: #ffffcc'] * len(row)
        else:
            return ['background-color: #ccffcc'] * len(row)

    st.dataframe(
        security_incidents.style.apply(highlight_severity, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Model integrity
    st.subheader("Model Integrity Checks")

    integrity_data = pd.DataFrame({
        'Check': ['Signature Verification', 'Tampering Detection', 'Encryption Status', 'Access Logs'],
        'Status': ['✅ PASSED', '✅ PASSED', '✅ ENCRYPTED', '✅ CLEAN'],
        'Last Verified': ['2026-03-20 00:00', '2026-03-20 00:00', '2026-03-20 00:00', '2026-03-20 00:00']
    })

    st.dataframe(integrity_data, use_container_width=True, hide_index=True)

    st.success("✅ All model integrity checks passed. No tampering detected.")

    st.markdown("---")

    # Blocked users
    st.subheader("Blocked Users (Active)")

    blocked_data = pd.DataFrame({
        'User ID': ['API_KEY_xyz123', 'API_KEY_abc789', 'IP_192.168.1.100'],
        'Reason': ['Honeypot trigger', 'Rate limit (hourly)', 'Rate limit (daily)'],
        'Blocked Until': ['2026-03-21 14:23', '2026-03-20 12:45', '2026-03-21 09:12'],
        'Abuse Score': [10.5, 2.0, 3.5]
    })

    st.dataframe(blocked_data, use_container_width=True, hide_index=True)

    st.warning("⚠️ **1 user** (API_KEY_xyz123) has high abuse score. Consider permanent ban.")

# Footer
st.markdown("---")
st.markdown("**Fraud Detection System v1.0** | All security features enabled ✅ | Auto-refresh: Manual")
