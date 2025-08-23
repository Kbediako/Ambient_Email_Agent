#!/usr/bin/env bash
set -e

ARTIFACTS_DIR="_artifacts"
LOG_FILE="$ARTIFACTS_DIR/s4_test_suite_logs.txt"

# Ensure artifacts dir exists
mkdir -p "$ARTIFACTS_DIR"

# Ensure src is on the python path
export PYTHONPATH="$(pwd)/src"

echo "INFO: Starting S4: Running full test suite..."

# Prefer a local virtualenv python if available
PYTHON_BIN="python3"
if [ -x ".venv_wsl/bin/python" ]; then
  PYTHON_BIN=".venv_wsl/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "--- Running Notebook Tests (pytest) ---" | tee -a "$LOG_FILE"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/test_notebooks.py -v 2>&1 | tee -a "$LOG_FILE"

echo -e "\n--- Running Python Tests (run_all_tests.py) ---" | tee -a "$LOG_FILE"
"$PYTHON_BIN" tests/run_all_tests.py 2>&1 | tee -a "$LOG_FILE"

echo -e "\nINFO: Test suite execution complete. Full log saved to $LOG_FILE"

