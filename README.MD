# Face Authentication Backend

A robust Flask-based facial recognition backend system that provides comprehensive face authentication services with support for single-image, multi-angle, and video-based recognition.

## Features

### Registration
- Single image student registration
- Multi-angle face registration (front, left, right angles)
- Batch registration support
- Quality-weighted face embedding generation

### Recognition
- Individual face recognition
- Group photo recognition
- Video-based recognition with temporal consistency
- Real-time face detection and matching

### Quality Controls
- Image quality validation
  - Brightness check
  - Blur detection
  - Resolution verification
- Face quality assessment
  - Pose estimation
  - Landmark detection
  - Size validation
- Video validation
  - Length limits
  - File size checks
  - Frame rate optimization

## Technical Stack

- **Framework**: Flask
- **Face Processing**: InsightFace
- **Image Processing**: OpenCV
- **Neural Computing**: NumPy
- **Similarity Matching**: Scikit-learn
- **Concurrent Processing**: ThreadPoolExecutor

## API Endpoints

### Registration Endpoints

#### 1. Single Image Registration
```http
POST /register_student
```
- Form data:
  - `image`: Student face image
  - `student_id`: Unique identifier

#### 2. Multi-angle Registration
```http
POST /register_student_multi_angle
```
- Form data:
  - `image_front`: Front face image
  - `image_left`: Left angle image
  - `image_right`: Right angle image
  - `student_id`: Unique identifier

#### 3. Batch Registration
```http
POST /batch_register
```
- Form data:
  - `images`: Multiple student images

### Recognition Endpoints

#### 1. Individual Recognition
```http
POST /recognize_student
```
- Form data:
  - `image`: Face image to recognize

#### 2. Group Recognition
```http
POST /recognize_group
```
- Form data:
  - `image`: Group photo

#### 3. Video Recognition
```http
POST /recognize_video
```
- Form data:
  - `video`: Video file

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FaceAuthBackend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
- The system will automatically create required directories
- InsightFace models will be downloaded on first run

5. Run the server:
```bash
python app.py
```
Server will start on port 6969

## Configuration

Key configurations in `config` dictionary:
- Image quality thresholds
- Face detection parameters
- Recognition similarity thresholds
- Video processing settings
- Cache sizes and temporal windows

## Error Handling

The system implements comprehensive error handling for:
- Invalid file types
- Poor image quality
- Multiple face detection
- Face angle validation
- Video processing issues
- Internal server errors

## Logging

- Detailed logging system
- Operation timestamps
- Error tracking
- Processing metrics

## Security Considerations

- File type validation
- Student ID format validation
- Image size limits
- Video length restrictions
- Error message sanitization

## Performance Optimization

- LRU caching for embeddings
- Multi-threaded video processing
- Quality-weighted matching
- Temporal consistency checks
- Configurable processing parameters
