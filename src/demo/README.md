# Interactive Fraud Detection Demos

## Overview

Two interactive web-based dashboards to demonstrate the fraud detection system:

1. **Analyst Dashboard** - Human-in-the-loop workflow for reviewing fraud alerts
2. **Monitoring Dashboard** - Real-time system health, fairness, drift, and security monitoring

---

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly pandas numpy
```

### 2. Run Analyst Dashboard

```bash
streamlit run src/demo/analyst_dashboard_streamlit.py
```

**Features:**
- ✅ Interactive alert queue with filters
- ✅ Alert detail view with customer info
- ✅ Model ensemble predictions visualization
- ✅ Human-in-the-loop decision making
- ✅ Victim support protocol for vulnerable populations
- ✅ Performance metrics tracking

**Demo Flow:**
1. See pending fraud alerts in priority order
2. Click on CRITICAL alert (elderly customer)
3. Review customer info + transaction details
4. See ensemble model predictions (3 models)
5. Review risk factors (reason codes)
6. Make decision (APPROVE/BLOCK/ESCALATE)
7. For vulnerable populations → victim support protocol triggered
8. Submit decision with notes
9. Alert moves to completed queue

### 3. Run Monitoring Dashboard

```bash
streamlit run src/demo/monitoring_dashboard_streamlit.py
```

**Features:**
- ✅ Real-time system health metrics
- ✅ Fairness & bias monitoring (80% rule, FPR parity)
- ✅ Drift detection (PSI scores, distribution plots)
- ✅ Security alerts & adversarial defense tracking
- ✅ Model integrity checks

**Tabs:**
- **System Health:** Transactions, alerts, queue status, performance
- **Fairness Monitoring:** Disparate impact testing, demographic FPR
- **Drift Detection:** Feature drift (PSI), prediction distribution
- **Security:** API rate limiting, honeypot triggers, blocked users

---

## Demo Screenshots

### Analyst Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  Fraud Alert Review Queue                    Analyst: analyst_001 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  🔴 CRITICAL - Alert FRD-001 - ¥5,000,000 - 3 min 🧓 Elderly     │
│  ├─ Customer: CUST-ELDERLY-001, Age: 72                          │
│  ├─ Balance Drain: 85% (CRITICAL)                                │
│  ├─ Model Scores: Balance: 0.94, Velocity: 0.88, Temporal: 0.91 │
│  ├─ Ensemble: 0.92 (CRITICAL)                                    │
│  ├─ Risk Factors: Phone scam pattern, IB recently enabled        │
│  └─ Decision: [APPROVE] [BLOCK] [REQUEST INFO] [ESCALATE]        │
│                                                                   │
│  🟠 HIGH - Alert FRD-002 - ¥1,200,000 - 15 min                   │
│  🟡 MEDIUM - Alert FRD-003 - ¥800,000 - 45 min                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  System Health | Fairness | Drift Detection | Security          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📊 Today's Metrics                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Transactions │ │ Fraud Alerts │ │ FPR: 4.2%    │            │
│  │   45,234     │ │     127      │ │ ▼ -0.3%      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                   │
│  ⚖️ Fairness Monitoring (80% Rule)                               │
│  Age Group:        ██████████████████ 0.87 ✅ PASS              │
│  Income Level:     ████████████████████ 0.92 ✅ PASS            │
│                                                                   │
│  📈 Drift Detection (PSI Scores)                                 │
│  balance_drain_ratio:  ██ 0.08  ✅ No drift                     │
│  amount_jpy:           ███ 0.12 ✅ Slight change                │
│                                                                   │
│  🔒 Security Incidents                                           │
│  • 14:23 Honeypot triggered - User blocked 24h                   │
│  • 11:45 Rate limit exceeded - User blocked 1h                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

### Demo 1: Protecting Elderly Scam Victim

**Scenario:** 72-year-old customer tricked by phone scammer

**Steps:**
1. Open Analyst Dashboard
2. See CRITICAL alert for elderly customer
3. Notice 🧓 VULNERABLE flag
4. Review: ¥5M transfer, 85% balance drain, phone scam pattern
5. Make decision: BLOCK
6. Victim support protocol auto-triggered:
   - ✅ Transaction blocked
   - ✅ Partial freeze (only outgoing transfers)
   - ✅ Customer notification (gentle language)
   - ✅ Victim resources sent
   - ⚠️ Required: Phone call within 30 min

**Key Features Demonstrated:**
- Human-in-the-loop (no automated blocking)
- Vulnerable population protection
- Victim support protocol
- Clear reason codes

### Demo 2: Fairness Monitoring

**Scenario:** Weekly bias audit

**Steps:**
1. Open Monitoring Dashboard
2. Navigate to "Fairness Monitoring" tab
3. See Disparate Impact Testing:
   - All groups pass 80% rule ✅
   - Age Group: 0.87 (≥ 0.80)
   - Income Level: 0.92 (≥ 0.80)
4. See FPR by demographics:
   - Elderly: 4.5%
   - Youngest: 3.8%
   - Max disparity: 0.7% (acceptable)

**Key Features Demonstrated:**
- Automated bias testing
- 80% rule compliance
- Equal opportunity (FPR parity)
- Vulnerable population tracking

### Demo 3: Drift Detection & Retraining

**Scenario:** Daily drift check

**Steps:**
1. Open Monitoring Dashboard
2. Navigate to "Drift Detection" tab
3. See Feature Drift (PSI scores):
   - balance_drain_ratio: 0.08 ✅ No drift
   - account_age_days: 0.18 🟡 Monitor closely
   - All features < 0.25 (retraining threshold)
4. See Prediction Distribution:
   - Training vs Production overlapping ✅
5. Recommendation: No retraining required

**Key Features Demonstrated:**
- Population Stability Index (PSI)
- Automatic drift detection
- Retraining triggers
- Distribution visualization

### Demo 4: Adversarial Attack Detection

**Scenario:** Attacker probing model

**Steps:**
1. Open Monitoring Dashboard
2. Navigate to "Security" tab
3. See Security Incidents:
   - 14:23 Honeypot triggered (user set _debug_mode = 1)
   - Action: User blocked 24 hours
   - Abuse score: 10.5 (high)
4. See Blocked Users table
5. Consider permanent ban for high abuse score

**Key Features Demonstrated:**
- Honeypot feature detection
- Rate limiting
- Abuse tracking
- Automatic blocking

---

## Customization

### Add Your Own Data

**Analyst Dashboard:**
Edit `analyst_dashboard_streamlit.py` line ~30:

```python
st.session_state.alerts = pd.DataFrame({
    # Your real alerts data here
})
```

**Monitoring Dashboard:**
Edit `monitoring_dashboard_streamlit.py` line ~25:

```python
@st.cache_data
def generate_demo_data():
    # Load your real production data here
    fraud_data = pd.read_csv('your_data.csv')
    return fraud_data
```

### Add Real-Time Updates

Add auto-refresh:

```python
import time

# In sidebar
auto_refresh = st.checkbox("Auto-refresh (30s)")

if auto_refresh:
    time.sleep(30)
    st.rerun()
```

---

## Production Deployment

### Using Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy both dashboards

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/demo/analyst_dashboard_streamlit.py"]
```

Build and run:

```bash
docker build -t fraud-analyst-dashboard .
docker run -p 8501:8501 fraud-analyst-dashboard
```

### Using AWS/Azure/GCP

Deploy as web service:
- AWS: Elastic Beanstalk or App Runner
- Azure: App Service
- GCP: Cloud Run

---

## Tips

### Performance
- Use `@st.cache_data` for expensive computations
- Load data once, cache in `st.session_state`
- Paginate large tables

### UX
- Use tabs to organize content
- Add filters for large datasets
- Show progress bars for long operations
- Use color coding (🔴 🟠 🟡 ⚪)

### Security
- Add authentication (Streamlit supports OAuth)
- Encrypt sensitive data
- Use HTTPS in production
- Rate limit API calls

---

## Troubleshooting

### Port already in use

```bash
streamlit run src/demo/analyst_dashboard_streamlit.py --server.port 8502
```

### Module not found

```bash
pip install streamlit plotly pandas numpy
```

### Dashboard not loading

Check console for errors:
```bash
streamlit run src/demo/analyst_dashboard_streamlit.py --logger.level=debug
```

---

## Next Steps

**Enhance Dashboards:**
- [ ] Add real-time WebSocket updates
- [ ] Integrate with production API
- [ ] Add user authentication (OAuth)
- [ ] Export reports to PDF
- [ ] Email/Slack notifications
- [ ] Mobile responsive design
- [ ] Dark mode support

**New Demos:**
- [ ] Customer appeal portal
- [ ] Victim support workflow
- [ ] Admin configuration panel
- [ ] Bias mitigation simulator

---

## Related Documentation

- [docs/final-report.md](../../docs/final-report.md) — Full technical report
- [docs/shinkaevolve.md](../../docs/shinkaevolve.md) — ShinkaEvolve integration and discoveries
