import os, io, tarfile, eumdac, requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import gzip, zipfile
import numpy as np
from pyresample import area_config
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from satpy import Scene
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

import struct

def read_nimrod_scaling(raw_header_bytes):
    """
    Parses the 512-byte NIMROD binary header to extract the exact
    dimensions, MKS scaling factor, data offset, units string, and
    the real-world British National Grid origin/spacing.
    """
    num_rows = struct.unpack('>h', raw_header_bytes[34:36])[0]
    num_cols = struct.unpack('>h', raw_header_bytes[36:38])[0]

    y_origin   = struct.unpack('>f', raw_header_bytes[74:78])[0]
    row_step   = struct.unpack('>f', raw_header_bytes[78:82])[0]
    x_origin   = struct.unpack('>f', raw_header_bytes[82:86])[0]
    column_step = struct.unpack('>f', raw_header_bytes[86:90])[0]

    mks_scaling = struct.unpack('>f', raw_header_bytes[94:98])[0]
    data_offset = struct.unpack('>f', raw_header_bytes[98:102])[0]
    units = raw_header_bytes[354:362].decode('ascii', errors='ignore').strip()

    print(f"[NIMROD Header] x_origin={x_origin}, y_origin={y_origin}, "
          f"col_step={column_step}, row_step={row_step}")

    return mks_scaling, data_offset, units, num_rows, num_cols, x_origin, y_origin, row_step, column_step


def get_ceda_data(target_year="", target_date="", target_time=""):
    """
    Fetches data from CEDA using the provided access token.
    Returns the data if successful, or None if there was an error.
    """

    key = (target_year, target_date)

    with ceda_lock:
        if key not in ceda_tar_cache:
            # Create a session container that tracks headers across requests. This is useful for maintaining authentication headers, cookies, and other session-related data across multiple requests to the same server.
            ceda_session = requests.Session()

            ceda_session.headers["Authorization"] = f"Bearer {ceda_token}"

            # The URL for CEDA's archive services that will be used to extract data. This URL points to a specific dataset in the CEDA archive.
            test_url = f"https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/{target_year}/metoffice-c-band-rain-radar_uk_{target_year}{target_date}_1km-composite.dat.gz.tar"

            # Use your authenticated session to send a fast test request
            response = ceda_session.get(test_url, stream=True)

            # Print the HTTP response code (200 means success, 401/403 means auth failed, 404 means not found, etc.)
            print(f"CEDA Connection Status: {response.status_code}")

            if response.status_code != 200:
                print(f"Failed to fetch data from CEDA: {response.status_code}")
                return None
            ceda_tar_cache[key] = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:")

        tar = ceda_tar_cache[key]
        file_name = f"metoffice-c-band-rain-radar_uk_{target_year}{target_date}{target_time}_1km-composite.dat.gz"
        member = tar.getmember(file_name)
        return tar.extractfile(member).read()



def get_eumetsat_token():
    """
    Fetches an access token from EUMETSAT using the provided consumer key and secret.
    Returns the access token if successful, or None if there was an error.
    """

    global eumetsat_token, eumetsat_token_expiry

    with token_lock:
        if eumetsat_token is not None and datetime.now() < eumetsat_token_expiry:
            return eumetsat_token
        
        eumetsat_key = os.environ.get("EUMETSAT_CONSUMER_KEY")
        eumetsat_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
        payload = {"grant_type": "client_credentials"}
        response = requests.post("https://api.eumetsat.int/token",auth=(eumetsat_key, eumetsat_secret), data=payload)
        
        credentials = (eumetsat_key, eumetsat_secret)

        if response.status_code == 200:
            eumetsat_token = eumdac.AccessToken(credentials)
            eumetsat_token_expiry = datetime.now() + timedelta(minutes=55)
            return eumetsat_token
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
    f = ceda_write_data(year, month, day, time_hrs, time_mins, time_secs)
    f = gzip.decompress(f)
    header = f[:512]
    raw_data = f[512:]
    dump_real_header(header)   # <-- add this line temporarily

    mks_scaling, data_offset, units, num_rows, num_cols, x_origin, y_origin, row_step, column_step = read_nimrod_scaling(header)
    print(f"\n[NIMROD Header] Dimensions: {num_rows}x{num_cols}, Scaling: {mks_scaling}, Units: '{units}'")

    data_array = np.frombuffer(raw_data, dtype=np.int16)
    return data_array, mks_scaling, data_offset, units, num_rows, num_cols, x_origin, y_origin, row_step, column_step

def dump_real_header(raw_header_bytes):
    """Print every float32 in the real header block with its index, so we
    can visually identify which slots actually hold x_origin/y_origin/
    row_step/column_step."""
    print("\n[Header Dump] Real header floats (index: value)")
    for i in range(28):
        offset = 66 + i * 4
        val = struct.unpack('>f', raw_header_bytes[offset:offset+4])[0]
        print(f"  idx {i:2d} (bytes {offset}-{offset+4}): {val}")



def load_tensor_data(year, month, day, time_hrs, time_mins, time_secs):

    print("--- Checking CEDA Spatial Array ---")
    t0 = time.time()
    ceda_data, mks_scaling, data_offset, units, num_rows, num_cols, x_origin, y_origin, row_step, column_step = load_nimrod_file(year, month, day, time_hrs, time_mins, time_secs)
    print(f"[timing] CEDA fetch+parse: {time.time()-t0:.2f}s")
    assert (num_rows, num_cols) == (2175, 1725), f"Unexpected grid shape: {num_rows}x{num_cols}"
    grid_data = ceda_data[:num_rows * num_cols].reshape(num_rows, num_cols)

    # Derive the true geographic extent safely using the top-down limits
    x_min = x_origin
    x_max = x_origin + (num_cols * column_step)
    
    # y_origin (1549500) represents the top edge (North maximum limit). 
    # To find the bottom edge, we subtract the total height from the top.
    y_top = y_origin
    y_bot = y_origin - (num_rows * row_step)


    # area_extent for pyresample: [xmin, ymin, xmax, ymax] -> [West, South, East, North]
    area_extent = [x_min, y_bot, x_max, y_top]
    
    # grid_extent for cartopy: [xmin, xmax, ymin, ymax] -> [West, East, South, North]
    grid_extent = [x_min, x_max, y_bot, y_top]

    assert all(np.isfinite(v) for v in area_extent), "area_extent contains NaN/Inf — check header byte offsets for x_origin/y_origin/row_step/column_step"

    
    print("\n--- Checking EUMETSAT Spatial Array ---")
    t1 = time.time()
    product = eu_write_data(year, month, day, time_hrs, time_mins, time_secs)
    print(f"[timing] EUMETSAT search: {time.time()-t1:.2f}s")

    # Creates a temporary directory that deletes itself automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the full path inside the temporary directory using a static name
        eumetsat_path = os.path.join(tmpdir, "eumetsat_data.zip")
        
        # Streams the data from EUMETSAT and write it to the temp folder
        t2 = time.time()
        with product.open() as f_in:
            with open(eumetsat_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"[timing] EUMETSAT download: {time.time()-t2:.2f}s")

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
            scn.load(['IR_108'])
            print("Raw IR_108 shape:", scn['IR_108'].shape)

            t_crop = time.time()
            scn = scn.crop(ll_bbox=(-13.47, 47.0, 4.0, 62.7))
            print(f"[timing] Crop: {time.time()-t_crop:.2f}s")
            print("Cropped IR_108 shape:", scn['IR_108'].shape)

            # Define the target CEDA British National Grid geometry
            area_id = 'bng'
            description = 'CEDA National Grid UK 1km'
            proj_id = 'bng'
            projection = {'init': 'epsg:27700'}
            width = num_cols
            height = num_rows

            target_area = area_config.get_area_def(area_id, description, proj_id, projection, width, height, area_extent)

            print("Target area definition created successfully.")

            print("--- Resampling EUMETSAT to CEDA Grid ---")
            t_resample = time.time()
            local_scn = scn.resample(target_area, resampler='nearest', cache_dir='/tmp/satpy_resample_cache')
            print(f"[timing] Resample: {time.time()-t_resample:.2f}s")

            # Extract the newly aligned IR_108 data as a numpy array
            eu_aligned_data = local_scn['IR_108'].values 
    """ 
    print("Aligned EUMETSAT shape:", eu_aligned_data.shape)

    print("Minimum temperature (Kelvin):", np.nanmin(eu_aligned_data))
    print("Maximum temperature (Kelvin):", np.nanmax(eu_aligned_data))

    print("Raw Radar Min:", np.nanmin(grid_data))
    print("Raw Radar Max:", np.nanmax(grid_data)) 
    
    """
    """ print("Raw radar percentiles:", np.nanpercentile(grid_data[grid_data >= 0], [50, 90, 99, 99.9]))
    print("Raw radar max:", grid_data.max())
    # Print the unique values that are SMALL (under 500) to see the real rain scale
    print("Sample of real rain values:", np.unique(grid_data[grid_data < 500])[-10:]) """
    
    # Convert raw int16 array to float32
    radar_cleaned = grid_data.astype(np.float32)
    # Identify error codes and system flags (values < 0 or >= 30000)
    radar_invalid = (grid_data < 0) | (grid_data >= 30000)
    radar_cleaned[radar_invalid] = np.nan
    # 1.0 means data is valid; 0.0 means it was a NaN or system flag
    valid_mask = (~radar_invalid) & (~np.isnan(eu_aligned_data))
    valid_mask = valid_mask.astype(np.float32)

    # Convert raw integers to physical rain rates using the official factor (divide by 32.0)
    radar_mmhr = radar_cleaned / 32.0

    # Use a realistic max ceiling for heavy rain (e.g., 24.0 mm/hr is an absolute downpour)
    max_physical_rain = 24.0
    radar_clipped = np.clip(radar_mmhr, 0.0, max_physical_rain)

    # Normalize linearly between 0.0 and 1.0 for the ML tensor
    norm_radar = radar_clipped / max_physical_rain
    
    # Preserve NaNs strictly for the Cartopy plot visualization background
    radar_plot = norm_radar.copy()
    radar_plot[radar_invalid] = np.nan
    

    # Min-Max normalization for satellite data, clipped to a valid [0,1] range
    norm_satellite = (eu_aligned_data - 200) / (310 - 200)
    norm_satellite = np.clip(norm_satellite, 0.0, 1.0)
    # Create a copy that keeps NaNs just for the visual plot
    satellite_plot = norm_satellite.copy()

    # Replace any NaNs resulting from normalization or missing data with 0.0
    norm_satellite[np.isnan(norm_satellite)] = 0.0
    norm_radar[np.isnan(norm_radar)] = 0.0

    # Re-stack the cleaned layers into our final tensor
    final_tensor = np.stack([norm_radar, norm_satellite], axis=0).astype(np.float16)
    print("Cleaned final tensor shape:", final_tensor.shape, final_tensor.dtype)

    # Save the Tensor to Disk 
    return final_tensor

    #visualize_data(norm_radar, satellite_plot, final_tensor, grid_extent)



def event_dataset(storm_windows, output_dir="data/training", max_workers=1):
    os.makedirs(output_dir, exist_ok=True)

    for start, end in storm_windows:
        event_id = f"event_{start:%Y%m%d_%H%M}_{end:%H%M}"
        output_path = f"{output_dir}/{event_id}.npy"
        if os.path.exists(output_path):
            print(f"Skipping {event_id}, already exists")
            continue

        timestamps = []
        ts = start
        while ts <= end:
            timestamps.append(ts)
            ts += timedelta(minutes=5)

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(load_tensor_data, f"{t.year}", f"{t.month:02d}", f"{t.day:02d}",
                                 f"{t.hour:02d}", f"{t.minute:02d}", "00"): t
                for t in timestamps
            }
            for future in as_completed(futures):
                t = futures[future]
                try:
                    results[t] = future.result()
                except Exception as e:
                    print(f"Failed frame {t}: {e}")

        frames = [results[t] for t in timestamps if t in results]   # keep original time order

        if frames:
            sequence = np.stack(frames, axis=0)
            np.save(output_path, sequence)
            print(f"Saved {event_id}: {sequence.shape}, {sequence.nbytes / 1e6:.1f} MB")
        else:
            print(f"No frames succeeded for {event_id}, skipping save")


# Visualises the data from a single tensor, showing the radar and satellite layers separately and overlaid, as well as their alignment on a map.
def visualize_data(norm_radar, satellite_plot, final_tensor, grid_extent):
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(norm_radar, cmap='Blues', alpha=0.6)
    ax.imshow(np.ma.masked_where(np.isnan(satellite_plot), satellite_plot), cmap='Reds', alpha=0.4)
    ax.set_title("Overlay check: radar coverage disc should sit inside land, "
             "not offset or flipped relative to satellite")
    plt.show()

    # Visualize the Aligned Layers
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Normalised Radar
    im1 = axes[0].imshow(final_tensor[0], cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title("Normalized Radar Reflectivity (CEDA)")
    fig.colorbar(im1, ax=axes[0], label="Scaled Intensity (0-1)")

    # Plot Normalised Satellite
    im2 = axes[1].imshow(final_tensor[1], cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title("Normalized Infrared Temperature (EUMETSAT)")
    fig.colorbar(im2, ax=axes[1], label="Scaled Temperature (0-1)")

    plt.tight_layout()
    plt.show()

    # Use new variable names (fig2, axes2) so it doesn't overwrite Figure 1
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 12), subplot_kw={'projection': ccrs.OSGB()}, dpi=120)

    # Verified real-world BNG extent for the UK 1km composite radar grid
    x0, x1, y0, y1 = grid_extent

    # Plot Radar on axes2[0]
    axes2[0].set_xlim(x0, x1)
    axes2[0].set_ylim(y0, y1)
    im1_map = axes2[0].imshow(
        final_tensor[0], cmap='YlOrRd', vmin=0, vmax=1,
        extent=grid_extent, transform=ccrs.OSGB(), origin='upper'
    )
    axes2[0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes2[0].set_title("Map Alignment: Radar Reflectivity")
    fig2.colorbar(im1_map, ax=axes2[0], label="Scaled Intensity (0-1)")

    # Plot Satellite on axes2[1]
    axes2[1].set_xlim(x0, x1)
    axes2[1].set_ylim(y0, y1)
    im2_map = axes2[1].imshow(
        final_tensor[1], cmap='inferno', vmin=0, vmax=1,
        extent=grid_extent, transform=ccrs.OSGB(), origin='upper'
    )
    axes2[1].add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1)
    axes2[1].set_title("Map Alignment: Infrared Temperature")
    fig2.colorbar(im2_map, ax=axes2[1], label="Scaled Temperature (0-1)")

    plt.tight_layout()
    plt.show()
    plt.close()



ceda_tar_cache = {}
eumetsat_token = None
eumetsat_token_expiry = None
DEBUG = False
ceda_lock = threading.Lock()
token_lock = threading.Lock()


# Define the storm windows for which we want to extract training data
storm_windows = [
    (datetime(2026, 7, 6, 13, 35), datetime(2026, 7, 6, 13, 45)),
]

event_dataset(storm_windows)



