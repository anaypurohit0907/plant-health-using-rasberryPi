# Plant Health Web Application

This project is a Flask web application that captures images from a camera, processes them to assess plant health using NDVI (Normalized Difference Vegetation Index), and displays the results on a web interface. If the camera is unavailable, the app automatically switches to a demo mode that rotates through images found in the `static/` directory.

## Project Structure

```
plant-health-web
├── app.py                 # Flask app with camera + demo-mode fallback and NDVI processing
├── requirements.txt       # Python dependencies
├── static/                # Place demo images here (png/jpg/jpeg); NDVI output saved here
├── templates/
│   └── index.html         # Web UI: live feed, capture button, health score, NDVI image
└── README.md              # Project docs
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd plant-health-web
   ```

2. **Install required packages**:
   Ensure you have Python 3.9+ installed, then install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the web application by running:
   ```
   python app.py
   ```

4. **Access the application**:
   Open your browser to http://localhost:5000

## Usage

- The Live Feed panel shows the camera stream. If the camera is not available, the app enters DEMO MODE and streams demo images from `static/`.
- Click "Capture & Analyze" to run contrast stretching and NDVI. The app saves `static/ndvi_image.png`, computes a 0–100 health score, and displays the result.
- Health categories: Poor (0–40), Fair (41–60), Good (61–80), Excellent (81–100).

## Demo mode notes

- Place at least one `.png`, `.jpg`, or `.jpeg` image of a plant in the `static/` folder to use demo mode.
- The app detects camera availability at startup and on any read failure. It will automatically switch to demo mode if needed and log the reason.
