# DeepFake Video and Image Detection

A deep learning-based project to detect deepfake images and videos using TensorFlow, Keras, and OpenCV.

## Features

- **Image Detection**: Detect deepfake images using a CNN model
- **Video Detection**: Analyze videos frame-by-frame for deepfake detection with conservative logic (any fake frame makes video fake)
- **Streamlit Web Interface**: User-friendly interface for uploading and analyzing media
- **Face Detection**: OpenCV-based face detection ensures analysis only occurs on valid faces
- **Pre-trained Model Support**: Includes both image and video detection models

## Installation

1. Clone this repository or download the project files

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
deepfake_detection/
├── README.md
├── requirements.txt
├── app.py                     # Streamlit web application
├── deepfake_image_model.h5    # Pre-trained image detection model
├── deepfake_video_model.h5    # Pre-trained video detection model
├── test_face_extraction.py    # Validation tests
├── model/
│   ├── __init__.py
│   └── deepfake_model.py      # Model architecture definitions
└── utils/
    ├── __init__.py
    ├── detection.py           # Core detection engine
    └── preprocessing.py       # Image/video preprocessing utilities
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Testing Face Detection

```bash
python test_face_extraction.py
```

## Detection Logic

- **Conservative Video Analysis**: If any frame in a video is detected as fake, the entire video is classified as fake
- **Face Detection Required**: Analysis only proceeds if faces are detected in the media
- **Model Types**: Automatic detection of image (CNN) vs video (LSTM+Xception) models

## Model Architecture

The project supports two model types:

### Image Model (CNN)
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- Binary classification (Real vs Fake)

### Video Model (LSTM + Xception)
- Frame-by-frame analysis using Xception backbone
- LSTM for temporal sequence modeling
- Conservative classification logic

## Dependencies

- TensorFlow 2.15.0
- Keras 2.15.0
- OpenCV
- Streamlit
- NumPy
- Pillow

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## License

MIT License
