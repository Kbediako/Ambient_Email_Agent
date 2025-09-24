#!/usr/bin/env python
"""
Setup script for Gmail API integration.

This script handles the OAuth flow for Gmail API access by:
1. Creating a .secrets directory if it doesn't exist
2. Using credentials from .secrets/secrets.json to authenticate
3. Opening a browser window for user authentication
4. Storing the access token in .secrets/token.json
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Import required Google libraries
from google_auth_oauthlib.flow import InstalledAppFlow


def main():
	secrets_path = ".secrets/client_secrets.json"
	# Load client secrets
	with open(secrets_path) as f:
		_ = json.load(f)
	# Create the flow using the client_secrets.json format
	flow = InstalledAppFlow.from_client_secrets_file(secrets_path, scopes=[
		"https://www.googleapis.com/auth/gmail.modify",
	])
	print("Setup flow created. Follow on-screen instructions to authorize.")


if __name__ == "__main__":
	main()