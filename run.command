#!/bin/bash
# Double-click in Finder (or run ./run.command) to launch the live dashboard.
# First run creates .venv and installs dependencies; later runs start instantly.
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up Python environment …"
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt || exit 1
fi

echo "Starting SPX IV dashboard at http://localhost:8501  (close this window to stop)"
exec .venv/bin/streamlit run app.py --server.headless false --browser.gatherUsageStats false
