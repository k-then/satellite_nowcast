# Satellite Nowcasting Project


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
