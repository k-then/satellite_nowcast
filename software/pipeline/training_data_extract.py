import os
from pathlib import Path
from dotenv import load_dotenv

# Get the current directory of this file
current_dir = Path(__file__).resolve().parent

# Navigate two levels up to get the repository root
repo_root = current_dir.parent.parent

# Create the path to the .env file in the repository root
env_path = repo_root / ".env"

# Load the environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# Fetch the keys to verify they aren't empty (None)
ceda_token = os.environ.get("CEDA_ACCESS_TOKEN")
eumetsat_key = os.environ.get("EUMETSAT_CONSUMER_KEY")

print(f"CEDA Token Found: {ceda_token is not None}")
print(f"EUMETSAT Key Found: {eumetsat_key is not None}")

import requests

# Create a session container that tracks headers across requests. This is useful for maintaining authentication headers, cookies, and other session-related data across multiple requests to the same server.
ceda_session = requests.Session()

ceda_session.headers["Authorization"] = f"Bearer {ceda_token}"

# The base URL for CEDA's archive services
test_url = "https://archive.ceda.ac.uk/"

# Use your authenticated session to send a fast test request
response = ceda_session.get(test_url)

# Print the HTTP response code (200 means success, 401/403 means auth failed)
print(f"CEDA Connection Status: {response.status_code}")

def get_eumetsat_token():
    """
    Fetches an access token from EUMETSAT using the provided consumer key and secret.
    Returns the access token if successful, or None if there was an error.
    """
    eumetsat_key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    eumetsat_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")

    payload = {
        "grant_type": "client_credentials"}
    
    response = requests.post(
        "https://api.eumetsat.int/token",
        auth=(eumetsat_key, eumetsat_secret), data=payload)
    
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Failed to get EUMETSAT token: {response.status_code} - {response.text}")
        return None

eumetsat_token = get_eumetsat_token()
print(f"EUMETSAT Connection Status: {'Success' if eumetsat_token else 'Failed'}")
