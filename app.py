from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import numpy as np
import os
import logging
import random

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Initialize the camera (will be validated below)
camera = cv2.VideoCapture(0)

# Demo mode state
DEMO_MODE = False
DEMO_IMAGES = []
_demo_index = 0

def _static_dir() -> str:
    return os.path.join(os.path.dirname(__file__), 'static')

def _init_demo_images() -> None:
    """Populate the list of demo images from the static directory."""
    global DEMO_IMAGES
    static_dir = _static_dir()
    if not os.path.isdir(static_dir):
        DEMO_IMAGES = []
        return
    candidates = [os.path.join(static_dir, f) for f in os.listdir(static_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # Sort by modification time (latest last) to have deterministic rotation
    DEMO_IMAGES = sorted(candidates, key=lambda p: os.path.getmtime(p))

def _ensure_camera_or_demo():
    """Validate camera availability; if unavailable, enable DEMO_MODE and load demo images."""
    global DEMO_MODE
    try:
        if not camera.isOpened():
            raise RuntimeError("Camera not opened")
        # Try one read to confirm
        ret, _ = camera.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        logging.info("Camera initialized successfully.")
    except Exception as e:
        logging.warning("Camera unavailable: %s. Switching to DEMO_MODE.", e)
        DEMO_MODE = True
        _init_demo_images()
        if not DEMO_IMAGES:
            logging.error("DEMO_MODE active but no demo images found in '%s'", _static_dir())

def read_image_from_camera():
    """Capture an image from the camera or return a demo image if DEMO_MODE is active.

    On camera read failure at runtime, automatically switch to DEMO_MODE.
    """
    global _demo_index, DEMO_MODE
    if DEMO_MODE:
        if not DEMO_IMAGES:
            raise RuntimeError("DEMO_MODE active but no demo images available in static/")
        # Rotate through demo images
        img_path = DEMO_IMAGES[_demo_index % len(DEMO_IMAGES)]
        _demo_index += 1
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read demo image: {img_path}")
        return img

    # Normal camera mode
    ret, frame = camera.read()
    if not ret or frame is None:
        logging.error("Camera read failed during runtime. Switching to DEMO_MODE.")
        DEMO_MODE = True
        _init_demo_images()
        if not DEMO_IMAGES:
            raise RuntimeError("Camera read failed and no demo images available in static/.")
        return read_image_from_camera()
    return frame

def contrast_stretch(im: np.ndarray, out_min: float = 0.0, out_max: float = 255.0,
                     in_low_percent: float = 5.0, in_high_percent: float = 95.0) -> np.ndarray:
    arr = np.asarray(im, dtype=float)
    low = np.percentile(arr, in_low_percent)
    high = np.percentile(arr, in_high_percent)
    if high == low:
        out = np.clip(arr, out_min, out_max)
    else:
        out = (arr - low) * ((out_max - out_min) / (high - low)) + out_min
        out = np.clip(out, out_min, out_max)
    return out.astype(np.uint8)

def calc_ndvi(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image.astype(float))
    denom = (r + b)
    eps = 1e-6
    ndvi = (r - b) / (denom + eps)
    ndvi_vis = ((ndvi + 1.0) / 2.0) * 255.0
    return np.clip(ndvi_vis, 0, 255).astype(np.uint8)

def compute_health_score(ndvi_vis: np.ndarray) -> dict:
    """Compute a simple 0–100 health score and category from NDVI visualization (0–255).

    Returns a dict with keys: score (int), label (str), mean_ndvi (float)
    """
    mean_val = float(np.mean(ndvi_vis))  # 0-255 scale
    score = int(np.clip((mean_val / 255.0) * 100.0, 0, 100))
    if score <= 40:
        label = "Poor"
    elif score <= 60:
        label = "Fair"
    elif score <= 80:
        label = "Good"
    else:
        label = "Excellent"
    return {"score": score, "label": label, "mean_ndvi": mean_val}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    try:
        frame = read_image_from_camera()
    except Exception as e:
        logging.exception("Capture failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

    contrasted = contrast_stretch(frame)
    ndvi_vis = calc_ndvi(contrasted)

    # Save the NDVI image
    static_dir = _static_dir()
    os.makedirs(static_dir, exist_ok=True)
    output_filename = 'ndvi_image.png'
    output_path = os.path.join(static_dir, output_filename)
    cv2.imwrite(output_path, ndvi_vis)

    # Compute health score
    health = compute_health_score(ndvi_vis)

    return jsonify({
        "ok": True,
        "message": "Image captured and processed.",
        "ndvi_image": url_for('static', filename=output_filename),
        "health": health,
        "demo_mode": DEMO_MODE
    }), 200

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = read_image_from_camera()
            except Exception as e:
                logging.exception("Video feed read failed: %s", e)
                break
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Validate camera on startup
_ensure_camera_or_demo()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        try:
            if camera:
                camera.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass