#!/usr/bin/env python

import pytest
import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--agent-module", 
        action="store", 
        default="email_assistant_hitl_memory_gmail",
        help="Specify which email assistant module to test"
    )

@pytest.fixture(scope="session")
def agent_module_name(request):
    """Return the agent module name from command line."""
    value = request.config.getoption("--agent-module")
    # Mirror into env so import-time code can read it if needed
    os.environ["PYTEST_AGENT_MODULE"] = value
    return value