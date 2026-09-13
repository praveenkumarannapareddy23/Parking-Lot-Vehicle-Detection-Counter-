# 🚗 Technical Documentation: Parking Lot & Traffic Vehicle Counter

## 1. Problem Statement

Automated parking lot monitoring and traffic flow analysis face several major technical challenges when using standard computer vision approaches:

1. **Overcounting Parked Vehicles**: Simple video counters re-count stationary parked cars in every single frame, resulting in hugely inflated, meaningless total counts.
2. **Detail Loss on Distant Vehicles**: Processing high-resolution lot imagery through standard fixed resolution (e.g. $640 \times 640$) turns distant vehicles into $15\text{px}$ ambiguous blobs that AI models fail to detect.
3. **Perspective Miscalculation**: Cameras mounted at angles cause vehicle centroids to fall outside marked parking bays, even though the vehicle's tires are resting inside the bay.
4. **False Positive ID Churn**: Low-confidence or flickering detections in video streams can trigger fake vehicle IDs if single-frame track appearances are counted immediately.

---

## 2. Acceptance Criteria

The system was engineered to satisfy the following requirements:

- **Multi-Class Vehicle Detection**: Detect and categorize four primary vehicle classes: `car`, `motorcycle`, `bus`, and `truck`.
- **Dual Analytical Metrics**: 
  - *Instantaneous Occupancy*: Real-time vehicle tally present in a single frame.
  - *Cumulative Unique Vehicles*: Total distinct vehicles passing through the scene over time using persistent tracking.
- **Custom Region of Interest (ROI)**: Restrict counting to user-defined polygon regions or parking zones with customizable anchor containment rules (`bottom`/tire anchor, `center`, `overlap`).
- **Dual User Interfaces**: Provide both an interactive Streamlit Web Dashboard (`app.py`) and a scriptable Command Line Interface (`detect_vehicles.py`).
- **Structured Data Export**: Automatically output annotated media (JPG/MP4), summary JSON records, and frame-by-frame CSV logs.
- **Zero Model Retraining**: Utilize pre-trained Ultralytics YOLO11 model weights out-of-the-box.

---

## 3. High-Level System Overview

The application takes raw images or video feeds from web uploads or local storage paths, runs AI inference through a pre-trained **YOLO11** model, tracks objects frame-to-frame with **ByteTrack**, filters results against user-defined ROI polygons, and outputs annotated media alongside detailed analytics reports.

```
┌──────────────────────────┐      ┌──────────────────────────┐
│   Web Dashboard (app.py) │  OR  │ CLI (detect_vehicles.py) │
└─────────────┬────────────┘      └────────────┬─────────────┘
              │                                │
              └───────────────┬────────────────┘
                              ▼
           ┌─────────────────────────────────────┐
           │ vehicle_counter/pipeline.py         │
           │ (Image & Video Orchestration)       │
           └──────────────────┬──────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│ vehicle_counter/      │           │ vehicle_counter/      │
│ detector.py           │           │ roi.py                │
│ (YOLO11 Inference)    │           │ (Geometric Filtering) │
└───────────┬───────────┘           └───────────┬───────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
           ┌─────────────────────────────────────┐
           │ vehicle_counter/annotate.py         │
           │ (OpenCV Graphics & Overlays)        │
           └──────────────────┬──────────────────┘
                              ▼
           ┌─────────────────────────────────────┐
           │ Outputs: JPG / MP4 / JSON / CSV     │
           └─────────────────────────────────────┘
```

---

## 4. Tools & Technology Stack

| Layer | Component | Technology | Role & Purpose |
|---|---|---|---|
| **Frontend** | Web Application | **Streamlit** ($\ge 1.49.0$) | Interactive browser UI for parameters & live media review |
| **Frontend** | Interactive ROI Canvas | **streamlit-image-coordinates** | Point-and-click polygon drawing on frame previews |
| **Frontend** | Styling System | **Vanilla CSS3** | Custom Dark Glassmorphism aesthetic, glowing cards, and pills |
| **Backend** | Core Language | **Python 3.10+** | Pipeline logic, CLI argument parsing, and data models |
| **Backend** | AI Framework | **Ultralytics YOLO11** | High-speed multi-class vehicle detection |
| **Backend** | Deep Learning Engine | **PyTorch** ($\ge 2.2.0$) | Model tensor computations (CUDA / Apple MPS / CPU) |
| **Backend** | Multi-Object Tracker | **ByteTrack** (`lapx`) | Assigns and maintains persistent vehicle IDs across frames |
| **Backend** | Vision & Math Engine | **OpenCV (Headless)** ($\ge 4.10.0$) | Image ingest, polygon test math, and visual annotation |
| **Backend** | Data & Export | **Pandas & NumPy** | Frame-by-frame data aggregation and CSV generation |
| **Backend** | Transcoding | **FFmpeg** | Converts output videos to web-compatible H.264 MP4 format |

---

## 5. AI Models Used (`yolo11m` Architecture)

The system defaults to **`yolo11m.pt` (YOLO11 Medium)** model weights.

```
       Model Complexity vs. Detection Accuracy
       
       Model        Parameters    Accuracy (mAP)    Recommended Use Case
       ─────────────────────────────────────────────────────────────────
       yolo11n.pt   2.6 M         43.5 %            Low-power edge devices
       yolo11s.pt   9.4 M         47.0 %            Fast real-time draft
    👉 yolo11m.pt   20.1 M        51.5 % (DEFAULT)  Best Parking Lot Accuracy
       yolo11l.pt   25.3 M        53.4 %            Heavy GPU servers
```

### Why `yolo11m` is Chosen:
- **Optimal Feature Extraction**: Provides superior precision for dense parking lots where vehicles are parked close together.
- **Occlusion Resilience**: Higher capacity neural layers detect vehicles partially blocked by light poles, trees, or adjacent trucks.
- **Small Object Recognition**: When paired with auto-resolution scaling (`imgsz="auto"`), `yolo11m` accurately identifies small distant cars.

---

## 6. Objects & Classes Used

The system maps COCO dataset class indices to vehicle targets:

```python
# Defined in vehicle_counter/config.py
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
```

### `Detection` Dataclass
Every detection across the system is encapsulated in an immutable `Detection` object (`vehicle_counter/detector.py`):

```python
@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box (tire ground contact point)."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)
```

---

## 7. System Architecture & Complete Process Flow

```
[ Input Source (Web Upload / Path) ]
               │
               ▼
[ 1. Image Preprocessing & Resolution Calculation (config.py) ]
  └── Calculates imgsz = ceil(max(h, w) / 32) * 32 (Capped [640, 1920])
               │
               ▼
[ 2. Pre-NMS Vehicle Detection & Tracking (detector.py) ]
  └── YOLO11 predict/track with classes=[2, 3, 5, 7] filter
               │
               ▼
[ 3. Region of Interest Polygon Test (roi.py) ]
  └── cv2.pointPolygonTest against Tire Anchor point (bottom-center)
               │
               ▼
[ 4. Pipeline Analytics & Tracking Churn Guard (pipeline.py) ]
  └── Filters tracks surviving < 3 frames; computes Peak & Unique counts
               │
               ▼
[ 5. OpenCV HUD & Bounding Box Rendering (annotate.py) ]
  └── Draws class-specific colors & translucent background summary cards
               │
               ▼
[ 6. Data Export & Transcoding ]
  └── Writes annotated JPG/MP4 (H.264), JSON summaries, and CSV reports
```

---

## 8. Step-by-Step Code Walkthrough (Important Code in Every File)

### Step 1: Configuration & Resolution Math (`vehicle_counter/config.py`)

This file manages global defaults, hardware selection, and resolution calculation:

```python
def resolve_imgsz(requested: int | str, frame_shape: tuple) -> int:
    """Calculates inference size, snapping to a multiple of 32."""
    if requested != "auto":
        return int(requested)

    long_side = max(frame_shape[0], frame_shape[1])
    snapped = int(math.ceil(long_side / IMGSZ_STEP) * IMGSZ_STEP)
    return max(IMGSZ_MIN, min(IMGSZ_MAX, snapped))

def pick_device(requested: str | None = None) -> str:
    """Auto-detects PyTorch execution device (MPS / CUDA / CPU)."""
    if requested:
        return requested
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"
```

---

### Step 2: AI Inference & Detection Wrapper (`vehicle_counter/detector.py`)

This file wraps the raw `ultralytics.YOLO` API and performs pre-NMS class filtering:

```python
class VehicleDetector:
    def __init__(self, model_path: str = config.DEFAULT_MODEL, device: str | None = None, ...):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.class_ids = sorted(class_ids or config.VEHICLE_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz_for(frame),
            device=self.device,
            classes=self.class_ids,  # Pre-NMS class filter
            verbose=False,
        )
        return self._to_detections(results[0])
```

---

### Step 3: Region of Interest Math (`vehicle_counter/roi.py`)

Handles geometric polygon containment tests using point-in-polygon math:

```python
def filter_detections(
    detections: list[Detection],
    roi: ROI | None,
    frame_width: int,
    frame_height: int,
    rule: str = "bottom",
) -> list[Detection]:
    if roi is None:
        return detections

    poly = roi.denormalize(frame_width, frame_height)
    kept: list[Detection] = []
    for d in detections:
        if rule == "bottom":
            pt = d.anchor  # Tire ground contact point
        elif rule == "center":
            pt = d.center
        
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            kept.append(d)
    return kept
```

---

### Step 4: Pipeline Execution Engine (`vehicle_counter/pipeline.py`)

Orchestrates single-image and video processing loops:

```python
# Video processing loop with ByteTrack ID survival filter
for det in raw_detections:
    if det.track_id is None:
        continue
    history = track_history.setdefault(det.track_id, {"cls_name": det.cls_name, "hits": 0})
    history["hits"] += 1
    
    # Track ID must survive for at least min_track_hits (e.g. 3 frames)
    if history["hits"] == min_track_hits:
        confirmed_unique_ids.add(det.track_id)
        confirmed_by_class[det.cls_name] = confirmed_by_class.get(det.cls_name, 0) + 1
```

---

### Step 5: Visual Renderer (`vehicle_counter/annotate.py`)

Draws bounding boxes, labels, and HUD overlay cards onto frames:

```python
def draw_detections(frame: np.ndarray, detections: list[Detection], ...) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        color = config.color_for(d.cls_id)
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        if label_mode != "none":
            cv2.putText(out, d.label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
```

---

### Step 6: Command Line Interface (`detect_vehicles.py`)

Provides CLI entry points for batch processing:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector = VehicleDetector(model_path=args.model, conf=args.conf, iou=args.iou)
    roi = ROI.load(args.roi) if args.roi else None
    
    if source_kind(args.source) == "image":
        res = process_image(args.source, detector, roi=roi, out_dir=args.out)
        print_counts("IMAGE COUNTS", res.counts, res.total)
    else:
        res = process_video(args.source, detector, roi=roi, out_dir=args.out)
        print_counts("UNIQUE VEHICLES", res.unique_by_class, res.unique_total)
```

---

### Step 7: Streamlit Web Dashboard (`app.py`)

Powers the dark glassmorphism web interface:

```python
# Radio selector card for source mode
source_mode_choice = st.radio(
    "Source Selector",
    ["📁 Bundled Samples", "📤 Upload a File"],
    index=0, horizontal=True, label_visibility="collapsed",
)

# Execution trigger
if st.button("▶ Run Detection", type="primary"):
    result = process_image(source_path, detector, roi=roi)
    render_kpi_cards(result.counts, result.total)
    st.image(bgr_to_rgb(cv2.imread(str(result.output_image))))
```

---

## 9. Output Analysis & Results Structure

### 1. JSON Output Schema (`outputs/parking_lot_summary.json`)
```json
{
  "source": "data/input/parking_lot.jpg",
  "model": "yolo11m.pt",
  "resolution": "1920x1080",
  "imgsz_inference": 1920,
  "elapsed_seconds": 0.342,
  "total_vehicles": 101,
  "counts_by_class": {
    "car": 88,
    "truck": 7,
    "motorcycle": 4,
    "bus": 2
  }
}
```

### 2. CSV Frame-by-Frame Tracking Log (`outputs/street_traffic_frames.csv`)
```csv
frame_idx,timestamp_s,occupancy_total,car,truck,bus,motorcycle,unique_total_so_far
0,0.00,14,12,1,0,1,14
1,0.03,15,13,1,0,1,15
2,0.07,14,12,1,0,1,15
```

---

## 10. Conclusion

The **Parking Lot & Traffic Vehicle Counter** provides a robust, modular, and high-accuracy solution for automated vehicle analytics:

1. **Accuracy-First AI**: Standardizing on **`yolo11m`** ensures exceptional vehicle detection in complex, dense parking lots.
2. **Perspective Correctness**: The **Tire Anchor Rule (`bottom`)** prevents false ROI counts caused by camera angles.
3. **Tracking Integrity**: **ByteTrack** with a 3-frame hit confirmation guard avoids duplicate counting and ID churn.
4. **Seamless Usability**: Offers both a modern **Dark Glassmorphism Web UI** and a fast **CLI tool**, ready for Cloud deployment out-of-the-box.
