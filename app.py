from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import numpy as np
import os
import logging
import random
import time

# Try to import Picamera2 (Raspberry Pi). If not available, we'll fall back to OpenCV.
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except Exception:
    Picamera2 = None  # type: ignore
    PICAMERA2_AVAILABLE = False

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Stream performance/config controls (override via env on Pi if needed)
STREAM_MAX_FPS = int(os.getenv('STREAM_MAX_FPS', '8'))            # throttle FPS to reduce CPU
STREAM_WIDTH = int(os.getenv('STREAM_WIDTH', '640'))               # resize width for stream
STREAM_JPEG_QUALITY = int(os.getenv('STREAM_JPEG_QUALITY', '70'))  # 0-100
STREAM_HEARTBEAT_SEC = float(os.getenv('STREAM_HEARTBEAT_SEC', '100'))  # send a keep-alive frame at least this often

# Initialize camera handles (actual setup in _ensure_camera_or_demo)
camera = None                 # OpenCV VideoCapture instance or None
picam2 = None                 # Picamera2 instance or None
USING_PICAM2 = False

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

def _init_picamera2() -> None:
    """Attempt to initialize Picamera2 for Raspberry Pi."""
    global picam2, USING_PICAM2
    if not PICAMERA2_AVAILABLE:
        raise RuntimeError("Picamera2 module not available")
    picam2 = Picamera2()
    # Configure a modest video resolution to limit CPU usage
    width = STREAM_WIDTH if STREAM_WIDTH else 640
    video_config = picam2.create_video_configuration(main={"size": (int(width), int(width * 3 / 4))})
    picam2.configure(video_config)
    picam2.start()
    # Try one capture to validate
    _ = picam2.capture_array()
    USING_PICAM2 = True
    logging.info("Picamera2 initialized successfully.")


def _init_opencv_camera() -> None:
    """Attempt to initialize OpenCV VideoCapture camera."""
    global camera, USING_PICAM2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("OpenCV camera not opened")
    # Try a read to confirm
    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("OpenCV camera read failed")
    camera = cap
    USING_PICAM2 = False
    logging.info("OpenCV VideoCapture initialized successfully.")


def _ensure_camera_or_demo():
    """Validate camera availability; prefer Picamera2 on Pi, else OpenCV; fallback to DEMO_MODE."""
    global DEMO_MODE
    # Try Picamera2 first if available
    try:
        if PICAMERA2_AVAILABLE:
            _init_picamera2()
            return
    except Exception as e:
        logging.warning("Picamera2 init failed: %s", e)

    # Fall back to OpenCV
    try:
        _init_opencv_camera()
        return
    except Exception as e:
        logging.warning("OpenCV camera init failed: %s", e)

    # Final fallback: DEMO_MODE
    logging.warning("No camera available. Switching to DEMO_MODE.")
    DEMO_MODE = True
    _init_demo_images()
    if not DEMO_IMAGES:
        logging.error("DEMO_MODE active but no demo images found in '%s'", _static_dir())

def read_image_from_camera():
    """Capture an image from the camera or return a demo image if DEMO_MODE is active.

    On camera read failure at runtime, automatically switch to DEMO_MODE.
    """
    global _demo_index, DEMO_MODE, USING_PICAM2
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
    try:
        if USING_PICAM2 and picam2 is not None:
            # Picamera2 returns RGB; convert to BGR for OpenCV processing
            rgb = picam2.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            if camera is None:
                raise RuntimeError("Camera not initialized")
            ret, frame = camera.read()
            if not ret or frame is None:
                raise RuntimeError("OpenCV camera read failed")
        return frame
    except Exception as e:
        logging.error("Camera read failed during runtime (%s). Switching to DEMO_MODE.", e)
        DEMO_MODE = True
        _init_demo_images()
        if not DEMO_IMAGES:
            raise RuntimeError("Camera read failed and no demo images available in static/.")
        return read_image_from_camera()


def _prep_frame_for_stream(frame: np.ndarray) -> np.ndarray:
    """Resize frame for streaming to reduce CPU/bandwidth."""
    if frame is None:
        return frame
    if STREAM_WIDTH and frame.shape[1] > STREAM_WIDTH:
        h, w = frame.shape[:2]
        scale = STREAM_WIDTH / float(w)
        new_size = (int(STREAM_WIDTH), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
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
    # Prebuild a tiny heartbeat JPEG to keep the stream alive if needed
    hb_img = np.zeros((2, 2, 3), dtype=np.uint8)
    _, hb_jpeg = cv2.imencode(
        '.jpg', hb_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(30, STREAM_JPEG_QUALITY))]
    )

    def generate():
        frame_interval = 1.0 / max(1, STREAM_MAX_FPS)
        last_yield = time.time()
        while True:
            loop_start = time.time()
            try:
                frame = read_image_from_camera()
                frame = _prep_frame_for_stream(frame)
                ok, jpeg = cv2.imencode(
                    '.jpg', frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)]
                )
                if not ok:
                    logging.error("JPEG encode failed.")
                    # Fall back to heartbeat
                    chunk = (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + hb_jpeg.tobytes() + b'\r\n')
                else:
                    chunk = (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                yield chunk
                last_yield = time.time()
            except GeneratorExit:
                logging.info("Client disconnected from video_feed.")
                break
            except BrokenPipeError:
                logging.info("Broken pipe in video_feed; stopping stream.")
                break
            except Exception as e:
                # Don't immediately drop the stream; yield a heartbeat and try again
                logging.warning("Video feed error: %s; sending heartbeat and continuing", e)
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + hb_jpeg.tobytes() + b'\r\n')
                    last_yield = time.time()
                except Exception:
                    break
            finally:
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                # Ensure periodic heartbeat to avoid idle timeouts by proxies
                if (time.time() - last_yield) > max(frame_interval, STREAM_HEARTBEAT_SEC):
                    try:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + hb_jpeg.tobytes() + b'\r\n')
                        last_yield = time.time()
                    except Exception:
                        break
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Advise proxies/servers not to buffer and to keep connection alive
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Connection'] = 'keep-alive'
    resp.headers['X-Accel-Buffering'] = 'no'  # for nginx
    return resp

# Validate camera on startup
_ensure_camera_or_demo()

if __name__ == '__main__':
    try:
        # Avoid debug/reloader to reduce duplicate processes and CPU on Pi
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    finally:
        try:
            if camera is not None:
                camera.release()
        except Exception:
            pass
        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass