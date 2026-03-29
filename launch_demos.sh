#!/bin/bash

# Launch Fraud Detection Demos
# Usage: ./launch_demos.sh [analyst|monitoring|both]

echo "======================================================================"
echo "  Fraud Detection System - Interactive Demos"
echo "======================================================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements-demo.txt
    echo ""
fi

# Get demo choice
DEMO=${1:-both}

case $DEMO in
    analyst)
        echo "🚀 Launching Analyst Dashboard..."
        echo "   URL: http://localhost:8501"
        echo ""
        streamlit run src/demo/analyst_dashboard_streamlit.py
        ;;

    monitoring)
        echo "🚀 Launching Monitoring Dashboard..."
        echo "   URL: http://localhost:8501"
        echo ""
        streamlit run src/demo/monitoring_dashboard_streamlit.py
        ;;

    both)
        echo "🚀 Launching BOTH dashboards..."
        echo "   Analyst Dashboard: http://localhost:8501"
        echo "   Monitoring Dashboard: http://localhost:8502"
        echo ""
        echo "Press Ctrl+C to stop both dashboards"
        echo ""

        # Launch analyst dashboard in background
        streamlit run src/demo/analyst_dashboard_streamlit.py --server.port 8501 &
        ANALYST_PID=$!

        # Wait a bit for first one to start
        sleep 3

        # Launch monitoring dashboard in background
        streamlit run src/demo/monitoring_dashboard_streamlit.py --server.port 8502 &
        MONITORING_PID=$!

        # Wait for both
        wait $ANALYST_PID $MONITORING_PID
        ;;

    *)
        echo "Usage: ./launch_demos.sh [analyst|monitoring|both]"
        echo ""
        echo "Options:"
        echo "  analyst     - Launch Analyst Dashboard only (port 8501)"
        echo "  monitoring  - Launch Monitoring Dashboard only (port 8501)"
        echo "  both        - Launch both dashboards (ports 8501 & 8502)"
        echo ""
        exit 1
        ;;
esac
