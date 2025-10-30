# Sentinel-2 Ship Chips — End-to-End Spec (AIS → S2 tiles → “Clay-ready” 256×256)

> Goal: build a repeatable pipeline that **collects current AIS vessel positions**, waits a few days, **pulls matching Sentinel-2 scenes**, and **chips 256×256 px tiles** centered on ships. Outputs are packaged and documented so they’re **ready for Clay** (consistent size, band order, normalization, and metadata).

This spec assumes you’ll **start from** a small “S2 chipper” codebase (e.g., a Rasterio/GDAL chipping script) and extend it with the modules below. If the referenced repo differs, treat this as the target behavior.

---

## 0) High-level architecture

```
+-------------------+            +---------------------+            +--------------------+
| AIS Collector     |  writes    | Acquisition Planner |  queries   | Sentinel-2 Search  |
|  (stream now)     +----------->+  (match AIS->S2)    +----------->+  (CDSE / SH APIs)  |
+-------------------+            +---------------------+            +--------------------+
         |                                 |                                     |
         v                                 v                                     v
   ais/ YYYYMMDD/                plan/ pending.csv                         scenes/ metadata
                                     |                                              |
                                     v                                              v
                               +----------------+                         +--------------------+
                               | S2 Chipper     |  reads plan+scenes      |  Cloud/Land masks  |
                               | (256×256 px)   +------------------------->| (SCL/QA/NDWI)     |
                               +----------------+                          +-------------------+
                                     |
                                     v
                         chips/, labels/, indexes/, QA/
```

- **Storage**: local FS or S3-compatible bucket.  
- **Scheduler**: cron/GitHub Actions/Systemd timer.  
- **Config**: `.env` + `config.yaml` (see §8).  
- **Idempotency**: every tile has a deterministic ID from `{mmsi, acq_time, lat, lon}`.

---

## 1) Dataset definition (what we produce)

### 1.1 Tile geometry
- **Pixel size**: `256 × 256`.  
- **Ground sampling distance (GSD)**: **10 m/px** (Sentinel-2 10 m bands).  
- **Footprint**: `~2.56 km × 2.56 km`, **centered on AIS point**.  
- **CRS**: use the scene’s native **UTM**; store CRS/transform in GeoTIFF.

### 1.2 Bands to request (best practice for ships)
Request the following **Sentinel-2 L2A** bands:

- **Core (10 m)** → *always present in final chips*  
  1. **B02 (Blue, 490 nm, 10 m)**  
  2. **B03 (Green, 560 nm, 10 m)**  
  3. **B04 (Red, 665 nm, 10 m)**  
  4. **B08 (NIR, 842 nm, 10 m)**

- **Auxiliary (20 m → resample to 10 m, bicubic)** → *optional channels that help QA & water/ship contrast*  
  5. **B11 (SWIR1, 1610 nm, 20 m → 10 m)**  
  6. **B12 (SWIR2, 2190 nm, 20 m → 10 m)**

- **Masks**  
  - **SCL** (Scene Classification Layer) for clouds/shadows/land  
  - **QA60** (cloud confidence) if available from provider  
  - Provide **derived masks**: `cloud_mask`, `land_mask`, `water_mask` (see §6).

**Recommended exported stacks**
- **RGBN** 4-band stack: `[B02,B03,B04,B08]` (10 m) → light, fast, widely usable.  
- **RGBN+SWIR** 6-band stack: `[B02,B03,B04,B08,B11r,B12r]` (SWIR resampled to 10 m) → best for robust training & cloud/foam handling.

### 1.3 Data types & scaling
- **Primary GeoTIFF**: `float32`, **top-of-atmosphere/BOA reflectance** in `[0, 1.0]`.  
- **Preview JPG/PNG** (optional): 8-bit stretched RGB for quicklooks (see §7.2).

### 1.4 File layout
```
dataset_root/
  chips/
    {split}/
      {mmsi}/
        {date}/
          {tile_id}.tif               # 4- or 6-band float32 GeoTIFF (COG if possible)
          {tile_id}_quicklook.jpg     # optional preview
          {tile_id}.json              # metadata (schema §1.5)
  labels/
    {split}/
      {mmsi}/{date}/{tile_id}_label.json
  indexes/
    tiles.parquet                     # per-chip index (one row per tile)
    ais_observations.parquet          # raw AIS snapshots kept long-term
  QA/
    reports/{date}.html               # daily QA summaries
```

(Truncated for brevity in this preview — full content continues with all sections as in the specification.)
