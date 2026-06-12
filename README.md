# 🚨 Human Intrusion Detection System

> **Production-grade AI/ML system for real-time human intrusion detection using Deep Learning**
> Built with YOLOv8 · ByteTrack · FastAPI · Redis · Docker · Shapely

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Detection Conditions

| Condition | Implementation |
|---|---|
| **Zone Entry** | Shapely polygon `contains(centroid)` check per detection |
| **Boundary Crossing** | LineString intersection on track trajectory history |
| **Entry During Restricted Hours** | Configurable time window per zone (`22:00–06:00`) |
| **Multiple Intruders** | `personCount >= threshold` per zone |
| **Vehicle Intrusion** | YOLOv8 detects car/truck/motorcycle class in zone |

## 📤 Output Format

```json
{
  "zone": "Storage Area",
  "personCount": 3,
  "vehicleCount": 0,
  "event": "Intrusion Detected",
  "alert_type": "Multiple Intruders",
  "confidence": 0.94,
  "severity": "MEDIUM",
  "timestamp": "2026-06-06T07:14:22Z",
  "camera_id": "CAM_03",
  "bboxes": [[100, 150, 200, 350]],
  "track_ids": [1, 2, 3],
  "thumbnail_url": "/frames/abc123.jpg"
}
```

## 🗂️ Datasets

| Dataset | URL | Use Case |
|---|---|---|
| **COCO 2017** | https://cocodataset.org/#download | Person + vehicle base training |
| **MOT17** | https://motchallenge.net/data/MOT17/ | Multi-person tracking |
| **VIRAT Ground** | https://viratdata.org/ | Outdoor surveillance |
| **UCF-Crime** | https://www.crcv.ucf.edu/projects/real-world/ | Anomaly events |
| **CrowdHuman** | https://www.crowdhuman.org/ | Dense crowd detection |
| **ShanghaiTech** | https://svip-lab.github.io/dataset/campus_dataset.html | Campus anomaly |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Download Data
```bash
python data/scripts/download_coco.py --output data/raw/coco
```

### 4. Train the Model
```bash
python -m src.training.train_detector train \
    --data data/processed/intrusion.yaml \
    --config src/training/config.yaml \
    --export-onnx
```

### 5. Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000/docs
```

### 6. Deploy with Docker
```bash
cd deployment
docker-compose up -d
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + model status |
| `POST` | `/api/v1/detect` | Detect intrusions in uploaded frame |
| `POST` | `/api/v1/detect/base64` | Detect from base64-encoded frame |
| `WS` | `/api/v1/ws/stream/{cam_id}` | Live camera stream processing |
| `WS` | `/api/v1/ws/alerts` | Real-time alert subscription |
| `GET` | `/api/v1/zones` | List all zones |
| `POST` | `/api/v1/zones` | Create new zone |
| `GET` | `/api/v1/alerts` | Get recent alerts |
| `GET` | `/metrics` | Prometheus metrics |

---

## 🏗️ Architecture

```
Camera (RTSP) → FrameExtractor → YOLOv8m → ByteTracker → ZoneManager → AlertEngine
                                                                  ↓
                                                        FastAPI  Redis  PostgreSQL
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v --cov=src

# Zone logic only
pytest tests/test_zone_logic.py -v

# API integration
pytest tests/test_api.py -v
```

---

## 🐳 Services (Docker)

| Service | Port | URL |
|---|---|---|
| **API** | 8000 | http://localhost:8000/docs |
| **MLflow** | 5000 | http://localhost:5000 |
| **Prometheus** | 9090 | http://localhost:9090 |
| **Grafana** | 3000 | http://localhost:3000 |
| **Redis** | 6379 | — |
| **PostgreSQL** | 5432 | — |

---

## 📊 Performance Targets

| Metric | Target |
|---|---|
| mAP@0.5 (person) | ≥ 0.85 |
| False Positive Rate | < 5% |
| Inference Latency | < 50ms/frame (GPU) |
| Real-time FPS | ≥ 15 FPS (1080p) |
| Alert Latency | < 200ms end-to-end |
