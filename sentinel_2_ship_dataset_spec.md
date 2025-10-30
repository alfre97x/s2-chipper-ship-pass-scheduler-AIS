# Sentinel-2 Ship Chips — Predictive End-to-End Spec (AIS → S2 tiles → “Clay-ready” 256×256)

> Goal: build a repeatable **predictive** pipeline that synchronizes **AIS vessel positions** with **Sentinel-2 overpasses**, ensuring that each collected AIS snapshot corresponds to a satellite acquisition window. The system waits for imagery availability, retrieves Sentinel-2 data, and produces **256×256 px tiles** centered on vessels — packaged, labeled, and documented for use in **Clay** or similar foundational models.

---

## 0) Updated High-level Architecture

```
+--------------------+        +-------------------------+       +---------------------+
| Overpass Predictor |        | AIS Collector (timed)   |       | Sentinel-2 Search   |
| (S2A/S2B planner)  +------->+   + Random ship sampler  +------>+   & Scene Matcher   |
+---------+----------+        +-----------+--------------+       +----------+----------+
          |                               |                                |
          v                               v                                v
  overpass_schedule/             ais/YYYYMMDD/                     scenes/ metadata
                                      |                                       |
                                      v                                       v
                              +------------------+                    +----------------+
                              | S2 Chipper       |   --> 256×256 px   | Cloud/Land QA  |
                              |  (GeoTIFFs)      |       chips        | & Labeling     |
                              +------------------+                    +----------------+
```

- **Predictive scheduling:** instead of collecting AIS randomly, the system anticipates Sentinel-2 passes.
- **Random ship sampling:** still supported for statistical diversity (e.g., collecting random global positions daily).
- **Storage:** local FS or S3-compatible bucket.
- **Scheduler:** cron, Airflow, or GitHub Actions.

---

## 1) Sentinel-2 Overpass Prediction

### 1.1 Overview
Sentinel-2A and Sentinel-2B are in **sun-synchronous orbits**, crossing the equator at ~10:30 AM local time. Each satellite covers the same area every **5 days**, offset so that together they revisit roughly every **2–3 days**.

To ensure AIS and imagery overlap, the system predicts **exact overpass times** for given areas (tiles or coordinates). These predictions define when the AIS collector should capture ship positions.

### 1.2 Data sources for overpass prediction

#### Option 1: ESA Acquisition Plans (recommended for most cases)
- Download from ESA SciHub:  
  👉 [https://scihub.copernicus.eu/userguide/Acquisition_Plans](https://scihub.copernicus.eu/userguide/Acquisition_Plans)
- Each CSV/KML entry includes: Satellite, Tile ID (e.g., `T30SUJ`), UTC acquisition time.
- Parse these to build a table of `(tile_id, satellite, start_time_utc, end_time_utc)`.

Example:
```
2025-11-01T10:23:11Z, S2A, Tile T30SUJ, Orbit 19431
```
This means: Tile **T30SUJ** will be imaged by **Sentinel-2A** at **10:23 UTC on Nov 1**.

#### Option 2: Orbit Propagation with TLEs (dynamic, code-based)
If you need flexibility beyond fixed tiles:
- Use **Celestrak TLEs**:  
  👉 [https://celestrak.org/NORAD/elements/sentinel.txt](https://celestrak.org/NORAD/elements/sentinel.txt)
- Use a library like `orbit-predictor` or `sgp4` to compute passes over any coordinate.

Example:
```python
from orbit_predictor.sources import get_predictor_from_tle_lines
from datetime import datetime, timedelta

s2a_tle = [
  "SENTINEL-2A",
  "1 40697U 15028A   24291.49583333  .00000023  00000-0  00000-0 0  9992",
  "2 40697  98.5656 103.9600 0001614  91.4290 268.7084 14.30823738526948"
]

predictor = get_predictor_from_tle_lines("S2A", s2a_tle)
passes = predictor.get_next_passes(lat=36.5, lon=-4.1, horizon=7)
```
This returns UTC timestamps for the next 7 days of overpasses.

### 1.3 Scheduler configuration

Add to your `config.yaml`:
```yaml
satellite_tracking:
  satellites: ["S2A", "S2B"]
  tle_source: "https://celestrak.org/NORAD/elements/sentinel.txt"
  prediction_days: 7
  overpass_window_minutes: 5
  min_elevation_deg: 10
  area_tiles: ["T30SUJ", "T31SDF", "T32TNS"]
```

### 1.4 Automation cycle

| Step | Description |
|------|--------------|
| **1️⃣ Predict** | Weekly job creates `plan/overpass_schedule.csv` (next 7 days of passes). |
| **2️⃣ Schedule AIS collectors** | For each tile, run a short AIS capture ±5 minutes around predicted time. |
| **3️⃣ Store AIS snapshot** | Save `{mmsi, lat, lon, tile_id, timestamp}` tagged to that overpass. |
| **4️⃣ Delay 2–3 days** | Wait for L2A availability, then fetch Sentinel-2 imagery for those tiles. |
| **5️⃣ Chip & label** | Create 256×256 tiles as before, now guaranteed to match overpasses. |

### 1.5 Example commands
```bash
python -m src.predict_overpasses --days 7 --out plan/overpass_schedule.csv
python -m src.collect_ais --schedule plan/overpass_schedule.csv --window-min 5
```

---

## 2) AIS Collection & Random Sampling

### 2.1 Predictive AIS capture
The collector listens to **AISStream (aisstream.io)** or another live feed. When the current UTC time falls within an overpass window, it:
1. Subscribes to AISStream for the relevant tile’s bounding box.
2. Collects all distinct MMSI within ±5 min of the overpass.
3. Saves them to `ais/YYYYMMDD/ais_overpass_{tile_id}.parquet`.

### 2.2 Random daily ship sampling
To maintain dataset diversity, run a parallel **random ship sampler**:
- Subscribe to AISStream globally or over large oceanic regions.
- Randomly pick **N distinct MMSI** per day (e.g., 1000) across the stream.

Example snippet:
```python
import random, time, json, websockets, asyncio

async def sample_random_ships(count=1000):
    url = "wss://stream.aisstream.io/v0/stream"
    ships = {}
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"APIKey": "YOUR_KEY", "BoundingBoxes": []}))
        while len(ships) < count:
            msg = json.loads(await ws.recv())
            ais = msg.get("Message", {}).get("PositionReport", {})
            if not ais: continue
            mmsi = msg.get("MetaData", {}).get("MMSI")
            ships[mmsi] = {
                "mmsi": mmsi,
                "lat": ais.get("Latitude"),
                "lon": ais.get("Longitude"),
                "timestamp": ais.get("TimeUTC")
            }
        return list(ships.values())

asyncio.run(sample_random_ships())
```
Store as:
```
ais/YYYYMMDD/ais_random_sample.parquet
```

You can also randomly pick a few **tiles globally** per day, then record all ships inside those tiles.

### 2.3 Combined operation
Your system can thus collect:
- **Predictive batches** (ships under Sentinel-2 at pass time)
- **Random batches** (ships worldwide for background diversity)

Both feed the same chipping and labeling pipeline.

---

## 3) Scene Search, Download, and Delay Timing

### 3.1 Delay timer
Wait **2–3 days** after each overpass to ensure Sentinel-2 **L2A** scenes are available.

```yaml
scheduler:
  ais_collect_interval_days: 1
  chipper_lag_days: 3           # Wait 3 days before downloading S2 scenes
  chipper_search_window_hours: 24
```

In cron:
```bash
# Predict overpasses weekly
0 06 * * MON python -m src.predict_overpasses --days 7

# Collect AIS around each predicted overpass
*/5 * * * * python -m src.collect_ais --active-schedule plan/overpass_schedule.csv

# Download and chip 3 days later
0 10 * * * python -m src.chipper --plan plan/overpass_schedule_$(date -d '3 days ago' +%Y%m%d).csv
```

### 3.2 Scene search parameters
- Product: `S2_L2A`
- Time window: ±24 h of predicted acquisition
- Cloud cover: ≤ 30%
- Preferred provider: **CDSE** or **SentinelHub**

---

## 4) Outputs (same as before)
- 256×256 GeoTIFFs (`float32`, reflectance 0–1)
- Metadata JSON with AIS + Sentinel-2 info
- Optional NPZ tensors and quicklook JPGs
- Full provenance and split indexes

---

## 5) Data Resources Summary

| Resource | Purpose | URL |
|-----------|----------|-----|
| ESA Acquisition Plans | Official overpass schedule | [ESA SciHub](https://scihub.copernicus.eu/userguide/Acquisition_Plans) |
| Celestrak Sentinel TLEs | Orbital data for programmatic prediction | [Celestrak Sentinel elements](https://celestrak.org/NORAD/elements/sentinel.txt) |
| Copernicus Data Space Ecosystem | Sentinel-2 imagery search/download | [https://dataspace.copernicus.eu](https://dataspace.copernicus.eu) |
| AISStream | Live AIS data (free WebSocket feed) | [https://aisstream.io](https://aisstream.io) |

---

## 6) Summary of Predictive Pipeline

1. **Predict next overpasses** for selected tiles using ESA orbits or TLEs.
2. **Schedule AIS collectors** around those times.
3. **Record ships** in those tiles ±5 min of overpass.
4. **Wait 2–3 days** → Sentinel-2 L2A data published.
5. **Download scenes** and chip 256×256 windows around AIS positions.
6. **Label**, **mask**, and **package** for Clay-ready training.

This predictive approach ensures **time-synchronized AIS–Sentinel-2 pairs**, minimizes wasted downloads, and produces a **clean, scalable maritime dataset** for model development.

