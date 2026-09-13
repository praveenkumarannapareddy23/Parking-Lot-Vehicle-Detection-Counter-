Project Walkthrough and Explanation Guide

Welcome to the Vehicle Counter and Traffic Analytics system walkthrough! This guide provides a simple, clear explanation of how the project is set up, how the AI model detects and tracks vehicles, and how the interactive web dashboard works.

--------------------------------------------------------------------------------

1. Quick Setup and Run Commands

Follow these easy steps to run the application on your computer:

Step 1: Open Terminal and Activate Environment
cd vehicle-counter

# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Fetch Sample Images and Videos
python tools/fetch_samples.py

Step 4: Run the Application
- Detect vehicles on a sample parking lot photo:
  python detect_vehicles.py --source data/input/parking_lot.jpg

- Track moving vehicles in a video:
  python detect_vehicles.py --source data/input/street_traffic.mp4

- Launch the Interactive Web Dashboard:
  streamlit run app.py

--------------------------------------------------------------------------------

2. How the System Works (Simple Explanation)

The application processes images or video streams through a 5-step pipeline:

[ Input Image / Video ]
           │
           ▼
[ 1. Image Preprocessing & Auto-Scaling ] ──► Adjusts resolution to catch small or distant cars
           │
           ▼
[ 2. YOLO11 Vehicle Detection ]          ──► Detects Cars, Motorcycles, Buses, and Trucks
           │
           ▼
[ 3. ByteTrack Vehicle Tracking ]        ──► Assigns unique IDs to distinct moving vehicles
           │
           ▼
[ 4. Region of Interest (ROI) Test ]     ──► Filters out cars outside specified parking zones
           │
           ▼
[ 5. Rendering & Output Export ]         ──► Saves annotated images, videos, JSON, and CSV reports

Explanation of the 5 Steps:
1. Input Reading: OpenCV reads the image or video frame.
2. AI Detection (`yolo11m` Model): We use the **`yolo11m.pt` (YOLO11 Medium)** model because it provides the best detection accuracy and vehicle precision in complex parking environments.
3. Vehicle Tracking: The ByteTrack multi-object tracker follows each vehicle from frame to frame. This prevents counting the same parked car multiple times in every frame.
4. Region of Interest (ROI): If an ROI polygon zone is drawn or specified, only vehicles whose bottom tires touch inside the drawn zone are counted.
5. Output Generation: The system draws color-coded bounding boxes, outputs a JSON summary file, generates frame-by-frame CSV reports, and displays real-time counts in the Streamlit web dashboard.

--------------------------------------------------------------------------------

3. Two Important Counting Metrics

When analyzing traffic or parking lots, the system calculates two distinct numbers:

1. Current Occupancy: 
   - Question it answers: "How many vehicles are present in this single frame right now?"
   - Best for: Parking lot capacity management and real-time space availability.

2. Total Unique Vehicles: 
   - Question it answers: "How many distinct vehicles passed through this road or gate over time?"
   - Best for: Traffic volume monitoring, highway analytics, and gate counters.

--------------------------------------------------------------------------------

4. System Components (Frontend and Backend)

Frontend (User Interface):
- Streamlit Dashboard (app.py): Provides a browser interface with controls for adjusting parameters, picking video/image inputs, and reviewing live metrics.
- Dark Glassmorphism Styling: Uses custom CSS for a dark theme with purple accent colors, glowing cards, and clean typography.
- Interactive ROI Drawer: Uses streamlit-image-coordinates to allow users to click directly on an image to draw custom counting zones.

Backend (AI and Engine):
- Model Wrapper (vehicle_counter/detector.py): Loads YOLO11 weights and extracts vehicle detection bounding boxes.
- Pipeline Orchestration (vehicle_counter/pipeline.py): Runs single-image detection or video tracking loops.
- ROI Geometry Engine (vehicle_counter/roi.py): Evaluates geometric point-in-polygon rules to filter vehicles.
- Annotation Renderer (vehicle_counter/annotate.py): Draws clean bounding boxes, vehicle class labels, and HUD overlays using OpenCV.

--------------------------------------------------------------------------------

5. Key Settings Explained Simply

| Setting | What it controls | Recommended Value | Why it matters |
|---|---|---|---|
| Confidence (--conf) | Minimum score required to accept a box | 0.25 | Higher values reduce false detections; lower values catch faint cars |
| NMS IoU (--iou) | Overlap threshold to remove duplicate boxes | 0.45 | Prevents multiple boxes from being drawn around the same vehicle |
| Inference Size (--imgsz) | Image resolution fed into the AI model | auto | Scaling resolution allows the AI to catch small or distant cars easily |
| Min Track Hits (--min-track-hits) | Minimum frames a track must persist | 3 | Prevents temporary flickering glitches from being counted as real vehicles |

--------------------------------------------------------------------------------

6. Repository File Guide

- app.py: Streamlit web dashboard application.
- detect_vehicles.py: Command Line Interface (CLI) entry point.
- requirements.txt: List of Python library dependencies.
- README.md: Project documentation and tech stack breakdown.
- vehicle_counter/: Core Python package containing detector, pipeline, geometry, and annotation modules.
- data/input/: Directory where input images and videos are stored.
- data/roi/: Directory for saved ROI polygon JSON files.
- outputs/: Output folder where annotated images, processed videos, and CSV/JSON reports are saved.

--------------------------------------------------------------------------------

7. Frequently Asked Questions

Q: Does this require model training?
A: No! The project uses pre-trained YOLO11 weights that recognize standard vehicles out of the box.

Q: Can I run this on CPU?
A: Yes! The system automatically detects whether you have a CUDA GPU, Apple Silicon MPS, or CPU, and runs smoothly on all platforms.

Q: How do I save the results?
A: Detections are automatically saved into the outputs/ folder, and download buttons are provided in the web dashboard for images, videos, JSON summaries, and CSV reports.
