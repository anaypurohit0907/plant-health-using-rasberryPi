from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

def read_image_from_camera():
    """Capture an image from the camera."""
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Could not read from camera.")
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    frame = read_image_from_camera()
    contrasted = contrast_stretch(frame)
    ndvi_vis = calc_ndvi(contrasted)
    
    # Save the NDVI image
    output_path = os.path.join('static', 'ndvi_image.png')
    cv2.imwrite(output_path, ndvi_vis)
    
    return Response("Image captured and processed.", status=200)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = read_image_from_camera()
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

camera.release()
cv2.destroyAllWindows()