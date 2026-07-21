import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from rasterio.transform import from_origin
from rasterio.features import rasterize
import pyproj
from pyproj import Transformer
from dotenv import load_dotenv
from rasterio.merge import merge
from pystac_client import Client
import math
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import planetary_computer
import os

load_dotenv()

# Visualises the data from a single tensor, showing the radar and satellite layers separately and overlaid, as well as their alignment on a map.
def visualise_data(mask_tensor, ter_tens, land_cover_matrix, grid_extent):
    fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.OSGB()}, dpi=120)

    # Verified real-world BNG extent for the UK 1km composite radar grid
    x0, x1, y0, y1 = grid_extent
    
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Overlays the mask tensor onto the map canvas
    im = ax.imshow(
        mask_tensor, 
        cmap='Greens',          # Color map (e.g., green for land)
        alpha=0.4,             # Transparency so the coastline is visible underneath
        extent=grid_extent,    # Aligns the pixel grid boundaries to the map
        origin='upper'         # Matches your top-down matrix structure
    )

    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.OSGB()}, dpi=120)

    # Verified real-world BNG extent for the UK 1km composite radar grid
    x0, x1, y0, y1 = grid_extent
    
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Overlays the mask tensor onto the map canvas
    im = ax.imshow(
        ter_tens, 
        cmap='terrain',          # Color map (e.g., green for land)
        alpha=0.4,             # Transparency so the coastline is visible underneath
        extent=grid_extent,    # Aligns the pixel grid boundaries to the map
        origin='upper'         # Matches your top-down matrix structure
    )


    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    plt.tight_layout()
    plt.show()
    plt.close()

    print("Unique land cover values:", np.unique(land_cover_matrix))

    fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.OSGB()}, dpi=120)
    x0, x1, y0, y1 = grid_extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Overlays the mask tensor onto the map canvas
    im = ax.imshow(
        land_cover_matrix, 
        cmap='terrain',          # Color map for land classes
        alpha=0.4,              # Transparency so the coastline is visible underneath
        extent=grid_extent,     # Aligns the pixel grid boundaries to the map
        origin='upper',          # Matches your top-down matrix structure
        transform=ccrs.OSGB()
    )

    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    plt.tight_layout()
    plt.show()
    plt.close()


def generate_mask_grid():
    # Dimensions from your specification
    num_rows, num_cols = 2175, 1725
    x_min, x_max, y_bot, y_top = -404500.0, 1320500.0, -625500.0, 1549500.0

    # Calculates pixel resolution (1000 meters per pixel)
    res_x = (x_max - x_min) / num_cols
    res_y = (y_top - y_bot) / num_rows
    
    # Defines the spatial transform matrix
    # Arguments: (west_edge, north_edge, pixel_width, pixel_height)
    transform = from_origin(x_min, y_top, res_x, res_y)

    # Extracts vector land geometries from Cartopy (defaulted in Lat/Lon)
    land_feature = cfeature.LAND.with_scale('50m')
    geoms_latlon = list(land_feature.geometries())

    # Transforms the entire OSGB grid into Latitude/Longitude
    osgb_crs = ccrs.OSGB()
    lon_lat_crs = ccrs.PlateCarree() # Standard Lat/Lon projection
    geoms_osgb = [osgb_crs.project_geometry(g, src_crs=lon_lat_crs) for g in geoms_latlon]

    # Applies the geometries onto the pixel grid instantly using the transform
    mask_tensor = rasterize(
        shapes=geoms_osgb,
        out_shape=(num_rows, num_cols),
        transform= transform,
        fill=0.0,       # Background value (sea)
        default_value=1.0,  # Value for inside the shapes (land)
        dtype=np.float16
    )

    output_folder = "data/training/static_layers"
    os.makedirs(output_folder, exist_ok=True)
    mask_path = os.path.join(output_folder, "sea_mask.npy")
    np.save(mask_path, mask_tensor.astype(np.float16))
    return mask_tensor


# Function to compute Web Mercator Tile X, Y
def latlon_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def generate_terrain_tensor():
    NUM_ROWS, NUM_COLS = 2175, 1725
    X_MIN, X_MAX, Y_BOT, Y_TOP = -404500.0, 1320500.0, -625500.0, 1549500.0

    RES_X = (X_MAX - X_MIN) / NUM_COLS
    RES_Y = (Y_TOP - Y_BOT) / NUM_ROWS

    # Converts OSGB36 Bounds to Latitude/Longitude for Tile Selection
    # We use standard numerical mapping to bypass the database lookup
    transformer_to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer_to_wgs.transform(X_MIN, Y_BOT)
    lon_max, lat_max = transformer_to_wgs.transform(X_MAX, Y_TOP)

    ZOOM = 6
    x_start, y_end = latlon_to_tile(lon_min, lat_max, ZOOM)
    x_end, y_start = latlon_to_tile(lon_max, lat_min, ZOOM)

    # Loops and builds the list of remote URL strings
    tile_urls = []
    for x in range(x_start, x_end + 1):
        for y in range(y_end, y_start + 1):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{ZOOM}/{x}/{y}.tif"
            tile_urls.append(url)


    # Opens and merges the raw Web Mercator arrays locally
    src_files = [rasterio.open(url) for url in tile_urls]
    mosaic, mosaic_transform = merge(src_files)
    for src in src_files:
        src.close()

    # Builds the grid coordinates of the source mosaic data
    mosaic_array = mosaic[0]
    src_height, src_width = mosaic_array.shape
    src_x = mosaic_transform[2] + np.arange(src_width) * mosaic_transform[0]
    src_y = mosaic_transform[5] + np.arange(src_height) * mosaic_transform[4]

    # Create an interpolator over the downloaded raw mosaic data
    # Using bounds_error=False handles any minor edge mismatches gracefully
    interp = RegularGridInterpolator((src_y, src_x), mosaic_array, method='linear', bounds_error=False, fill_value=0)

    # Generates exact target OSGB36 coordinates
    target_x = np.linspace(X_MIN + RES_X/2, X_MAX - RES_X/2, NUM_COLS)
    target_y = np.linspace(Y_TOP - RES_Y/2, Y_BOT + RES_Y/2, NUM_ROWS)
    gx, gy = np.meshgrid(target_x, target_y)

    # Converts target OSGB coordinates to Web Mercator (EPSG:3857) to sample the mosaic
    transformer_to_mercator = Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True)
    merc_x, merc_y = transformer_to_mercator.transform(gx, gy)

    # Samples the pixels directly into your target matrix as float16
    sample_points = np.stack([merc_y.ravel(), merc_x.ravel()], axis=-1)
    elevation_matrix = interp(sample_points).reshape(NUM_ROWS, NUM_COLS).astype(np.float16)

    output_folder = "data/training/static_layers"
    os.makedirs(output_folder, exist_ok=True)
    terrain_path = os.path.join(output_folder, "terrain.npy")
    np.save(terrain_path, elevation_matrix.astype(np.float16))

    return elevation_matrix


def generate_land_type():
    # Target OSGB36 Grid Parameters
    X_MIN, X_MAX = -404500.0, 1320500.0
    Y_MIN, Y_MAX = -625500.0, 1549500.0
    NUM_ROWS, NUM_COLS = 2175, 1725
    
    # Converts OSGB36 bounds to WGS84 for the STAC API query
    transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer_to_wgs84.transform(X_MIN, Y_MIN)
    lon_max, lat_max = transformer_to_wgs84.transform(X_MAX, Y_MAX)
    wgs84_bbox = [lon_min, lat_min, lon_max, lat_max]

    # Queries STAC Catalog Items
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = client.search(collections=["esa-worldcover"], bbox=wgs84_bbox)
    items = list(search.item_collection())
    print(f"Found {len(items)} matching WorldCover tiles.")

    if not items:
        print("No tiles found.")
        return None

    # Signs items to obtain valid Azure asset access tokens
    tile_urls = [planetary_computer.sign_item(item).assets['map'].href for item in items]
    
    # Target grid coordinates for projection matching
    target_x = np.linspace(X_MIN, X_MAX, NUM_COLS)
    target_y = np.linspace(Y_MAX, Y_MIN, NUM_ROWS)
    gx, gy = np.meshgrid(target_x, target_y)
    tgt_lon, tgt_lat = transformer_to_wgs84.transform(gx, gy)
    
    # Opens and merges only the bounding box area from the cloud assets
    src_files = [rasterio.open(url) for url in tile_urls]
    
    # Calculate the resolution in degrees to match our target grid dimensions
    res_lon = (lon_max - lon_min) / NUM_COLS
    res_lat = (lat_max - lat_min) / NUM_ROWS

    # Merge and downsample simultaneously using the 'res' argument
    mosaic, mosaic_transform = merge(
        src_files, 
        bounds=(lon_min, lat_min, lon_max, lat_max),
        res=(res_lon, res_lat)
    )
    
    for src in src_files:
        src.close()

    # Builds grid coordinates of the cropped source mosaic data
    
    mosaic_array = mosaic[0]
    src_height, src_width = mosaic_array.shape
    src_x = mosaic_transform[2] + np.arange(src_width) * mosaic_transform[0]
    src_y = mosaic_transform[5] + np.arange(src_height) * mosaic_transform[4]

    # Creates the nearest-neighbor interpolator for discrete classes
    interp = RegularGridInterpolator((src_y, src_x), mosaic_array, method='nearest', bounds_error=False, fill_value=0)

    # Samples points directly into the target matrix
    sample_points = np.stack([tgt_lat.ravel(), tgt_lon.ravel()], axis=-1)
    land_cover_matrix = interp(sample_points).reshape(NUM_ROWS, NUM_COLS).astype(np.uint8)

    output_folder = "data/training/static_layers"
    os.makedirs(output_folder, exist_ok=True)
    land_path = os.path.join(output_folder, "land_type.npy")
    np.save(land_path, land_cover_matrix.astype(np.uint8))

    return land_cover_matrix

def get_wgs_bbox(x_min, x_max, y_bot, y_top):
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    # Convert all 4 corners: NW, NE, SW, SE
    lons, lats = transformer.transform(
        [x_min, x_max, x_min, x_max], 
        [y_top, y_top, y_bot, y_bot]
    )
    return min(lons), min(lats), max(lons), max(lats)


# Inside generate_land_type() and generate_terrain_tensor():
uk_extent = [-404500.0, 1320500.0, -625500.0, 1549500.0]
# lon_min, lat_min, lon_max, lat_max = get_wgs_bbox(uk_extent)
# wgs84_bbox = [lon_min, lat_min, lon_max, lat_max]




visualise_data(generate_mask_grid(), generate_terrain_tensor(), generate_land_type(), uk_extent)







