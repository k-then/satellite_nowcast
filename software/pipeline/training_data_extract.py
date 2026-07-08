import os, io, tarfile, eumdac, requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Get the current directory of this file
current_dir = Path(__file__).resolve().parent

# Navigate two levels up to get the repository root
repo_root = current_dir.parent.parent

# Create the path to the .env file in the repository root
env_path = repo_root / ".env"

# Load the environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# Fetch the keys to verify they aren't empty 
ceda_token = os.environ.get("CEDA_ACCESS_TOKEN")
eumetsat_key = os.environ.get("EUMETSAT_CONSUMER_KEY")

print(f"CEDA Token Found: {ceda_token is not None}")
print(f"EUMETSAT Key Found: {eumetsat_key is not None}")

def get_ceda_data(target_year="", target_date="", target_time=""):
    """
    Fetches data from CEDA using the provided access token.
    Returns the data if successful, or None if there was an error.
    """
    # Create a session container that tracks headers across requests. This is useful for maintaining authentication headers, cookies, and other session-related data across multiple requests to the same server.
    ceda_session = requests.Session()

    ceda_session.headers["Authorization"] = f"Bearer {ceda_token}"

    # The URL for CEDA's archive services that will be used to extract data. This URL points to a specific dataset in the CEDA archive.
    test_url = f"https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/{target_year}/metoffice-c-band-rain-radar_uk_{target_year}{target_date}_1km-composite.dat.gz.tar"

    # Use your authenticated session to send a fast test request
    response = ceda_session.get(test_url, stream=True)

    # Print the HTTP response code (200 means success, 401/403 means auth failed, 404 means not found, etc.)
    print(f"CEDA Connection Status: {response.status_code}")

    if response.status_code == 200:
        # Wrap the raw response stream so tarfile can read it
        tar_stream = io.BytesIO(response.content) 

        with tarfile.open(fileobj=tar_stream, mode="r:") as tar:
            # gets appropriate file name from the tar file
            file_name = f"metoffice-c-band-rain-radar_uk_{target_year}{target_date}{target_time}_1km-composite.dat.gz"
            member = tar.getmember(file_name) # Extract the file to the current working directory
            extracted_file = tar.extractfile(member) # open the extracted file as a file-like object
            output_file = extracted_file.read() # read the contents of the extracted file
            return output_file # return the contents of the extracted file
    else:
        print(f"Failed to fetch data from CEDA: {response.status_code}")
        return None



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
    
    credentials = (eumetsat_key, eumetsat_secret)

    if response.status_code == 200:
        return eumdac.AccessToken(credentials)
    else:
        print(f"Failed to get EUMETSAT token: {response.status_code} - {response.text}")
        return None

def get_eumetsat_data(start="", end=""):
    """
    Fetches data from EUMETSAT using the provided access token.
    Returns the data if successful, or None if there was an error.
    """
    search_args = {
        'geo': [47.0, -13.47, 62.7, 4.0],
        'dtstart': start,
        'dtend': end
    }

    # Fetch the EUMETSAT token and initialize the DataStore
    eumetsat_token = get_eumetsat_token()
    print(f"EUMETSAT Connection Status: {'Success' if eumetsat_token else 'Failed'}")
    datastore = eumdac.DataStore(eumetsat_token)
    collection = datastore.get_collection('EO:EUM:DAT:MSG:MSG15-RSS')

    results = collection.search(**search_args)
    return results


def write_data(year="", month="", day="", time_hrs="", time_mins="", time_secs=""):
    folder_path = "data/raw" # Specify the folder path where you want to save the data
    date = f"{month}{day}"  # Format the date as MMDD
    time = f"{time_hrs}{time_mins}"
    ceda_data = get_ceda_data(year, date, time)  # Example target year and time, adjust as needed
    start_dt = datetime(int(year), int(month), int(day), int(time_hrs), int(time_mins), int(time_secs))
    end_dt = start_dt + timedelta(minutes=5)  
    eu_data = get_eumetsat_data(f'{start_dt.year:02d}-{start_dt.month:02d}-{start_dt.day:02d}T{start_dt.hour:02d}:{start_dt.minute:02d}:{start_dt.second:02d}', f'{end_dt.year:02d}-{end_dt.month:02d}-{end_dt.day:02d}T{end_dt.hour:02d}:{end_dt.minute:02d}:{end_dt.second:02d}')

    # Writes the CEDA data to a file in the specified folder with a name that includes the year, date, and time.
    ceda_filename = f"ceda_{year}{date}_{time}.dat.gz" 
    with open(os.path.join(folder_path, ceda_filename), "wb") as f:
        f.write(ceda_data)

    # Writes the EUMETSAT data to a file in the specified folder with a name that includes the year, date, and time.
    product = next(iter(eu_data))
    eu_filename = f"eu_{year}{date}_{time}.dat.gz" 
    with product.open() as f_in, open(os.path.join(folder_path, eu_filename), "wb") as f_out:
        f_out.write(f_in.read())

    


# Usage of the write_data function with specific year, date, and time values. Adjust these values as needed.
year = "2026"
month = "06"
day = "15"
time_hrs = "14"
time_mins = "00"
time_secs = "00"
write_data(year, month, day, time_hrs, time_mins, time_secs)



