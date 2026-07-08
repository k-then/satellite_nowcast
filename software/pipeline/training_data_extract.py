import os, io, tarfile, eumdac, requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import gzip, zipfile
import numpy as np
import satpy
from pyresample import area_config
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from satpy import Scene
import tempfile

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


def ceda_write_data(year="", month="", day="", time_hrs="", time_mins="", time_secs=""):
    date = f"{month}{day}"  # Format the date as MMDD
    time = f"{time_hrs}{time_mins}"
    ceda_data = get_ceda_data(year, date, time)  # Example target year and time, adjust as needed
    return ceda_data  # Return the CEDA data for further processing or saving


def eu_write_data(year="", month="", day="", time_hrs="", time_mins="", time_secs=""):
    start_dt = datetime(int(year), int(month), int(day), int(time_hrs), int(time_mins), int(time_secs))
    end_dt = start_dt + timedelta(minutes=5)  
    eu_data = get_eumetsat_data(f'{start_dt.year:02d}-{start_dt.month:02d}-{start_dt.day:02d}T{start_dt.hour:02d}:{start_dt.minute:02d}:{start_dt.second:02d}', f'{end_dt.year:02d}-{end_dt.month:02d}-{end_dt.day:02d}T{end_dt.hour:02d}:{end_dt.minute:02d}:{end_dt.second:02d}')

    # Writes the EUMETSAT data to a file in the specified folder with a name that includes the year, date, and time.
    product = next(iter(eu_data))
    return product  # Return the EUMETSAT product for further processing or saving




def load_nimrod_file(year, month, day, time_hrs, time_mins, time_secs):
    """
    Loads a NIMROD file and returns its contents as a numpy array.
    """
    f = ceda_write_data(year, month, day, time_hrs, time_mins, time_secs)
    f = gzip.decompress(f)
    header = f[:512]       # Grabs everything from index 0 up to 512
    raw_data = f[512:]     # Grabs everything from index 512 to the very end
    data_array = np.frombuffer(raw_data, dtype=np.int16)  # Convert the raw data to a numpy array
    return data_array


def load_tensor_data(year, month, day, time_hrs, time_mins, time_secs):

    print("--- Checking CEDA Spatial Array ---")
    ceda_data = load_nimrod_file(year, month, day, time_hrs, time_mins, time_secs)
    grid_data = ceda_data[:3751875].reshape(2175, 1725)

    print("\n--- Checking EUMETSAT Spatial Array ---")
    product = eu_write_data(year, month, day, time_hrs, time_mins, time_secs)

    # Creates a temporary directory that deletes itself automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the full path inside the temporary directory using a static name
        eumetsat_path = os.path.join(tmpdir, "eumetsat_data.zip")
        
        # Streams the data from EUMETSAT and write it to the temp folder
        with product.open() as f_in:
            with open(eumetsat_path, 'wb') as f_out:
                f_out.write(f_in.read())
                
        with zipfile.ZipFile(eumetsat_path, 'r') as z:
            # Find the file name that ends with .nat
            nat_filename = [f for f in z.namelist() if f.endswith('.nat')][0]
            
            # Extract it to your data folder
            z.extract(nat_filename, path=tmpdir)
            temp_nat_path = Path(tmpdir) / nat_filename
            print(f"Extracted: {nat_filename}")

            # Open the zip file and list its contents
            print("Files inside EUMETSAT zip:")
            print(z.namelist())

            print("--- Loading EUMETSAT Native File ---")
            scn = Scene(reader="seviri_l1b_native", filenames=[temp_nat_path])

            # See what channels are available in this dataset
            print("Available channels:")
            print(scn.available_dataset_names())
            scn.load(['IR_108'])
            print("Raw IR_108 shape:", scn['IR_108'].shape)

            # Define the target CEDA British National Grid geometry
            area_id = 'bng'
            description = 'CEDA National Grid UK 1km'
            proj_id = 'bng'
            projection = {'init': 'epsg:27700'} # EPSG code for British National Grid
            width = 1725
            height = 2175
            area_extent = [-200000, -200000, 1525000, 1975000] # Bounds in meters

            target_area = area_config.get_area_def(area_id, description, proj_id, projection, width, height, area_extent)
            print("Target area definition created successfully.")

            print("--- Resampling EUMETSAT to CEDA Grid ---")
            # Resample the satellite scene to our target British National Grid area
            local_scn = scn.resample(target_area)

            # Extract the newly aligned IR_108 data as a numpy array
            eu_aligned_data = local_scn['IR_108'].values
    
    """ 
    print("Aligned EUMETSAT shape:", eu_aligned_data.shape)

    print("Minimum temperature (Kelvin):", np.nanmin(eu_aligned_data))
    print("Maximum temperature (Kelvin):", np.nanmax(eu_aligned_data))

    print("Raw Radar Min:", np.nanmin(grid_data))
    print("Raw Radar Max:", np.nanmax(grid_data)) 
    
    """

    # Mask out the invalid flags (set them to NaN)
    radar_masked = np.where((grid_data >= 0) & (grid_data <= 32000), grid_data, np.nan)

    # Max-Min normalise the valid weather data (0.0 to 1.0)
    norm_radar = (radar_masked - 0) / (32000 - 0)
    radar_plot = norm_radar.copy()

    # Min-Max normalization for satellite data
    norm_satellite = (eu_aligned_data - 200) / (310 - 200)
    # Create a copy that keeps NaNs just for the visual plot
    satellite_plot = norm_satellite.copy()

    # Replace any NaNs resulting from normalization or missing data with 0.0
    norm_satellite[np.isnan(norm_satellite)] = 0.0
    norm_radar[np.isnan(norm_radar)] = 0.0

    # Re-stack the cleaned layers into our final tensor
    final_tensor = np.stack([norm_radar, norm_satellite], axis=0)
    print("Cleaned final tensor shape:", final_tensor.shape)

    # Save the Tensor to Disk 
    output_path = f"data/training/{year}_{month}_{day}_{time_hrs}{time_mins}{time_secs}.npy"
    np.save(output_path, final_tensor)
    print(f"Successfully saved tensor to {output_path}")

    """ # Visualize the Aligned Layers
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Normalized Radar
    im1 = axes[0].imshow(final_tensor[0], cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title("Normalized Radar Reflectivity (CEDA)")
    fig.colorbar(im1, ax=axes[0], label="Scaled Intensity (0-1)")

    # Plot Normalized Satellite
    im2 = axes[1].imshow(satellite_plot, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title("Normalized Infrared Temperature (EUMETSAT)")
    fig.colorbar(im2, ax=axes[1], label="Scaled Temperature (0-1)")

    plt.tight_layout()
    plt.show()

    # Use new variable names (fig2, axes2) so it doesn't overwrite Figure 1
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.OSGB()})

    grid_extent = [-200000, 1525000, -200000, 1975000] 

    # Plot Radar on the new axes2[0]
    im1_map = axes2[0].imshow(radar_plot, cmap='YlOrRd', vmin=0, vmax=1, extent=grid_extent, origin='lower')
    axes2[0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes2[0].set_title("Map Alignment: Radar Reflectivity")
    fig2.colorbar(im1_map, ax=axes2[0], label="Scaled Intensity (0-1)")

    # Plot Satellite on the new axes2[1]
    im2_map = axes2[1].imshow(satellite_plot, cmap='inferno', vmin=0, vmax=1, extent=grid_extent, origin='lower')
    axes2[1].add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1)
    axes2[1].set_title("Map Alignment: Infrared Temperature")
    fig2.colorbar(im2_map, ax=axes2[1], label="Scaled Temperature (0-1)")

    plt.tight_layout()
    plt.show()
    plt.close() """


# Usage of the write_data functions with specific year, date, and time values. Adjust these values as needed.
year = "2026"
month = "06"
day = "15"
time_hrs = "14"
time_mins = "00"
time_secs = "00"
load_tensor_data(year, month, day, time_hrs, time_mins, time_secs)

