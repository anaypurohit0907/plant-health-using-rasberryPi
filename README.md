# Plant Health Web Application

This project is a simple web application that captures images from a camera, processes them to assess plant health using NDVI (Normalized Difference Vegetation Index) calculations, and displays the results on a web interface.

## Project Structure

```
plant-health-web
├── app.py                # Main Python application
├── templates
│   └── index.html       # HTML structure for the web application
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd plant-health-web
   ```

2. **Install required packages**:
   Ensure you have Python installed, then install the necessary packages using pip:
   ```
   pip install opencv-python flask numpy
   ```

3. **Run the application**:
   Start the web application by running:
   ```
   python app.py
   ```

4. **Access the application**:
   Open your web browser and navigate to `http://localhost:5000` to access the application.

## Usage

- Click the "Capture Image" button to take a photo using your camera.
- The application will process the image to assess plant health and display the results on the web page.
