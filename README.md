# Satellite Nowcasting Project

# UK Satellite-Radar Nowcasting Pipeline
This pipeline processes and aligns real-time meteorological observations over the UK to build high-resolution datasets for weather nowcasting. It automatically handles the spatial warping, normalisation, and temporal stacking required to feed raw satellite and radar inputs into deep learning models.



## Key Features

* **Spatial Regridding:** Automatically warps and resamples EUMETSAT geostationary satellite imagery to match the flat 1km CEDA radar grid exactly.
* **Physical Normalization:** Scales raw physical values (Kelvin and dBZ) to a standard [0, 1] range optimized for neural networks.
* **Temporal Stacking:** Stacks sequential frames back-to-back to construct 4D spatiotemporal tensors.
* **Hardware-Accelerated Pipeline:** Designed to leverage NVIDIA CUDA-enabled GPUs for heavy training and preprocessing. 
  * *Note:* Windows users should run this pipeline inside **WSL2 (Windows Subsystem for Linux)** for native CUDA support.
* **Edge-Ready for FPGAs:** Preprocessed tensors are structured to allow direct deployment for low-latency inference on hardware-accelerated edge devices like FPGAs.


## Grid & Spatial Alignment Specifications

* **Target Projection:** British National Grid (`OSGB36` / `EPSG:27700`)
* **CEDA Radar Grid Shape:** `(2175, 1725)`
* **Raw EUMETSAT (`IR_108`) Shape:** `(1392, 3712)` → **Aligned Shape:** `(2175, 1725)`


## Environment & Prerequisites

* **Python:** `3.13+`
* **Geospatial Processing:** `satpy`, `pyresample`, `pyproj`
* **Tensor Handling:** `numpy`
* **Static Layer Retrieval:** `rasterio`, `pystac_client`
* **Hardware Acceleration:** NVIDIA CUDA Toolkit (via WSL2 on Windows)


## Directory Structure

```text
satellite_nowcast/
├── 📁 checkpoints_v6/
├── 📁 data/
│   ├── 📁 raw/
│   └── 📁 training/
│       ├── 📁 dynamic_layers/
│       └── 📁 static_layers/
├── 📁 env_satellite/
├── 📁 forecast_output_v6/
│   ├── 🖼️ forecast_comparison.png
│   └── 📄 storm_summary.json
├── 📁 hardware/
│   ├── 📁 rtl/
│   └── 📁 tb/
├── 📁 software/
│   ├── 📁 model/
│   │   ├── 📁 __pycache__/
│   │   ├── 📁 checkpoints/
│   │   ├── 🐍 advection.py
│   │   ├── 🐍 ai_weather_model.py
│   │   ├── 🐍 dataset.py
│   │   ├── 🐍 inference.py
│   │   ├── 🐍 storm_tracking.py
│   │   └── 🐍 training.py
│   └── 📁 pipeline/
│       ├── 📁 __pycache__/
│       ├── 🐍 static_tensor.py
│       └── 🐍 training_data_extract.py
├── ⚙️ .env
├── 📄 .gitignore
├── 📜 Miniconda3-latest-Linux-x86_64.sh
└── 📄 README.md
```  
---
## Data Attributions & Citations

This project utilises a combination of real-time meteorological observations and static geographic datasets to train the nowcasting model.
### Dynamic Weather Datasets

1. **EUMETSAT MSG SEVIRI Rapid Scan Service (RSS)**
   * **Dataset Identifier:** `EO:EUM:DAT:MSG:MSG15-RSS`
   * **Sensor/Reader:** Spinning Enhanced Visible and Infrared Imager (`seviri_l1b_native`)
   * **Source:** [EUMETSAT Data Store](https://data.eumetsat.int/)
   * **Citation:** > EUMETSAT (2026). Meteosat Meteorological Product: SEVIRI Level 1.5 Image Data - Rapid Scan Service - Meteosat - 0 degree / Indian Ocean / Rapid Scan. Darmstadt, Germany: EUMETSAT.

2. **Met Office Nimrod UK 1km Rainfall Radar Composite**
   * **Dataset Directory:** `/badc/ukmo-nimrod/data/composite/uk-1km/`
   * **Source:** [Centre for Environmental Data Analysis (CEDA) Archive](https://catalogue.ceda.ac.uk/)
   * **Citation:**
     > Met Office (2026). Met Office Nimrod UK 1km Resolution Rainfall Radar Composite. Centre for Environmental Data Analysis.

### Static Geographic Datasets

3. **Land Cover / Land Use**
   * **Source:** [ESA WorldCover 10m product](https://esa-worldcover.org/) accessed via the [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/).
   * **Citation:** > ESA WorldCover 10 m 2020 / 2021 database © ESA WorldCover consortium.

4. **Topography (DEM)**
   * **Source:** [Mapzen Terrain Tiles](https://github.com/tilezen/joerd/blob/master/docs/attribution.md) hosted on AWS.
   * **Dataset:** A composite of global elevation datasets including SRTM, GMTED2010, and EUDEM.

5. **Coastline & Land Geometries**
   * **Source:** [Natural Earth](https://www.naturalearthdata.com/) vector maps (50m scale) compiled via `Cartopy`.
