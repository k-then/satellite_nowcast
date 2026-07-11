import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import contains_xy, prepare
import numpy as np
from rasterio.transform import from_origin
from rasterio.features import rasterize

# Visualises the data from a single tensor, showing the radar and satellite layers separately and overlaid, as well as their alignment on a map.
def visualise_data(mask_tensor, grid_extent):
    # Use new variable names (fig2, axes2) so it doesn't overwrite Figure 1
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


def generate_mask_grid():
    # Dimensions from your specification
    num_rows, num_cols = 2175, 1725
    x_min, x_max, y_bot, y_top = -404500.0, 1770500.0, -175500.0, 1549500.0

    # Calculates pixel resolution (1000 meters per pixel)
    res_x = (x_max - x_min) / num_cols
    res_y = (y_top - y_bot) / num_rows
    
    # Defines the spatial transform matrix
    # Arguments: (west_edge, north_edge, pixel_width, pixel_height)
    transform = from_origin(x_min, y_top, res_x, res_y)

    # Generates 1D coordinate arrays (pixel centers)
    x_coords = np.linspace(x_min + 500, x_max - 500, num_cols)
    y_coords = np.linspace(y_top - 500, y_bot + 500, num_rows) # Top-down for 'upper' origin

    # Creates the full 2D meshgrid of OSGB coordinates
    X_osgb, Y_osgb = np.meshgrid(x_coords, y_coords)

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
        transform=transform,
        fill=0.0,       # Background value (sea)
        default_value=1.0,  # Value for inside the shapes (land)
        dtype=np.float16
    )

    return mask_tensor


uk_extent = [-404500.0, 1770500.0, -175500.0, 1549500.0]
visualise_data(generate_mask_grid(), uk_extent)
