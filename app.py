from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import logging
from functools import lru_cache
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List , Generator
import re
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time

# Initialize Flask app
app = Flask(__name__)

# Configure InsightFace model download directory
os.environ['INSIGHTFACE_HOME'] = os.path.expanduser('~/.insightface')
model_dir = os.path.join(os.environ['INSIGHTFACE_HOME'], 'models')
os.makedirs(model_dir, exist_ok=True)

# Configuration
config = {
    'UPLOAD_FOLDER': 'uploads',
    'EMBEDDING_FOLDER': 'embeddings',
    'SIMILARITY_THRESHOLD': 0.5,

    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MIN_IMAGE_SIZE': 224,
    'MIN_BRIGHTNESS': 40,
    'MAX_BRIGHTNESS': 250,
    'MAX_CACHE_SIZE': 1000
}


# Add configuration for video processing
config.update({
    'MAX_VIDEO_LENGTH_SECONDS': 60,
    'VIDEO_PROCESSING_FPS': 2,
    'ALLOWED_VIDEO_EXTENSIONS': {'mp4', 'avi', 'mov'},
    'MAX_VIDEO_SIZE_MB': 50
})

# Add to config dictionary
config.update({
    'FACE_ANGLE_THRESHOLD': 30,  # Maximum allowed face angle in degrees
    'MIN_FACE_SCORE': 0.9,       # Minimum face detection confidence
    'MIN_FACE_SIZE': 80,         # Minimum face size in pixels
    'TEMPORAL_WINDOW': 5,        # Number of frames for temporal consistency
    'MIN_RECOGNITION_OCCURRENCES': 3  # Minimum occurrences for reliable recognition
})

# Add to config dictionary
config.update({
    'REQUIRED_ANGLES': ['front', 'left', 'right'],
    'ANGLE_THRESHOLDS': {
        'front': {'yaw': (-15, 15), 'pitch': (-15, 15)},
        'left': {'yaw': (30, 60), 'pitch': (-15, 15)},
        'right': {'yaw': (-60, -30), 'pitch': (-15, 15)}
    }
})

# Add new config parameters
config.update({
    'IMAGE_STANDARDIZATION': {
        'TARGET_SIZE': (640, 640),
        'MIN_FACE_SIZE': 160,
        'BLUR_THRESHOLD': 100,  # Laplacian variance threshold
        'CONTRAST_LIMITS': (0.3, 1.0),
        'BRIGHTNESS_RANGE': (0.8, 1.2),
    },
    'FACE_QUALITY': {
        'MIN_QUALITY_SCORE': 0.6,
        'MIN_SHARPNESS': 50,
        'MAX_HEAD_POSE': 30,
        'MIN_FACE_CONFIDENCE': 0.98,
    },
    'RECOGNITION': {
        'MIN_EMBEDDING_SIMILARITY': 0.6,
        'TEMPORAL_WINDOW_SIZE': 5,
        'MIN_SEQUENCE_LENGTH': 3,
    }
})

# Setup logging
logging.basicConfig(
    filename='face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create required directories
for folder in [config['UPLOAD_FOLDER'], config['EMBEDDING_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Initialize FaceAnalysis
def init_face_analyzer(use_gpu=False):
    try:
        # Initialize with either GPU or CPU
        ctx_id = 0 if use_gpu else -1
        app = FaceAnalysis(providers=['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider'])
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info(f"Successfully initialized FaceAnalysis with {'GPU' if use_gpu else 'CPU'}")
        return app
    except Exception as e:
        if use_gpu:
            logger.warning(f"GPU initialization failed: {str(e)}. Falling back to CPU.")
            return init_face_analyzer(use_gpu=False)
        else:
            logger.error(f"Face analyzer initialization failed: {str(e)}")
            raise

try:
    # Initialize face analyzer with CPU (more stable)
    face_analyzer = init_face_analyzer(use_gpu=False)
except Exception as e:
    logger.error(f"Error initializing face analyzer: {str(e)}")
    raise

def validate_student_id(student_id: str) -> bool:
    """
    Validate student ID format.
    """
    if not student_id or not isinstance(student_id, str):
        return False
    # Add your specific student ID format validation
    pattern = r'^[A-Za-z0-9]{5,20}$'  # Example pattern
    return bool(re.match(pattern, student_id))

def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config['ALLOWED_EXTENSIONS']

def check_image_quality(img: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Check if the image meets quality requirements.
    """
    # Check image size
    if img.shape[0] < config['MIN_IMAGE_SIZE'] or img.shape[1] < config['MIN_IMAGE_SIZE']:
        return False, "Image resolution too low"
        
    # Check brightness
    brightness = np.mean(img)
    if brightness < config['MIN_BRIGHTNESS'] or brightness > config['MAX_BRIGHTNESS']:
        return False, "Image too dark or too bright"
        
    return True, None

def process_multiple_faces(img: np.ndarray) -> Tuple[Optional[dict], Optional[str]]:
    """
    Process and validate face detection results.
    """
    faces = face_analyzer.get(img)
    if len(faces) > 1:
        return None, "Multiple faces detected"
    elif len(faces) == 0:
        return None, "No face detected"
    return faces[0], None

@lru_cache(maxsize=config['MAX_CACHE_SIZE'])
def get_stored_embedding(student_id: str) -> Optional[np.ndarray]:
    """
    Get stored embedding for a student with caching.
    """
    embedding_path = os.path.join(config['EMBEDDING_FOLDER'], f'{student_id}.json')
    try:
        with open(embedding_path, 'r') as f:
            stored_data = json.load(f)
            return np.array(stored_data['embedding'])
    except Exception as e:
        logger.error(f"Error loading embedding for student {student_id}: {str(e)}")
        return None

def validate_face_angle(face: dict, angle_type: str) -> bool:
    """
    Validate if face angle matches required pose.
    """
    try:
        pitch, yaw, roll = face.get('pose', [0, 0, 0])
        threshold = config['ANGLE_THRESHOLDS'][angle_type]
        
        return (threshold['yaw'][0] <= yaw <= threshold['yaw'][1] and 
                threshold['pitch'][0] <= pitch <= threshold['pitch'][1])
    except:
        return False

def combine_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple embeddings into a single enhanced embedding.
    """
    # Average the embeddings
    combined = np.mean(embeddings, axis=0)
    # Normalize the resulting embedding
    return combined / np.linalg.norm(combined)

def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Enhanced image preprocessing pipeline.
    """
    info = {'original_size': img.shape}
    
    # Standardize resolution
    target_size = config['IMAGE_STANDARDIZATION']['TARGET_SIZE']
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
        info['resized'] = True
    
    # Convert to LAB color space for better brightness/contrast adjustment
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    info['enhanced'] = True
    
    return img, info

def assess_image_quality(img: np.ndarray) -> Tuple[float, dict]:
    """
    Comprehensive image quality assessment.
    """
    quality_scores = {}
    
    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    quality_scores['sharpness'] = min(laplacian_var / config['FACE_QUALITY']['MIN_SHARPNESS'], 1.0)
    
    # Contrast assessment
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    quality_scores['contrast'] = min(contrast / 127, 1.0)
    
    # Overall quality score
    overall_score = np.mean(list(quality_scores.values()))
    
    return overall_score, quality_scores

def align_face(img: np.ndarray, landmarks: dict) -> np.ndarray:
    """
    Align face based on detected landmarks.
    """
    if landmarks is None:
        return img
        
    # Extract eye coordinates
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate angle and scale
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    scale = 1
    
    # Get rotation matrix
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply transformation
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    return aligned_img

def enhance_face_detection(img: np.ndarray) -> List[Dict]:
    """
    Enhanced face detection with multiple confidence checks.
    """
    faces = face_analyzer.get(img)
    enhanced_faces = []
    
    for face in faces:
        quality_score, quality_details = assess_image_quality(
            img[int(face['bbox'][1]):int(face['bbox'][3]), 
                int(face['bbox'][0]):int(face['bbox'][2])]
        )
        
        if quality_score < config['FACE_QUALITY']['MIN_QUALITY_SCORE']:
            continue
            
        # Enhance with quality metrics
        face['quality_score'] = quality_score
        face['quality_details'] = quality_details
        
        # Add pose estimation confidence
        if 'pose' in face:
            pitch, yaw, roll = face['pose']
            if max(abs(pitch), abs(yaw), abs(roll)) > config['FACE_QUALITY']['MAX_HEAD_POSE']:
                continue
        
        enhanced_faces.append(face)
    
    return enhanced_faces

def get_enhanced_embedding(img: np.ndarray, face: dict) -> np.ndarray:
    """
    Generate enhanced face embedding with preprocessing.
    """
    # Align face
    if 'landmarks' in face:
        img = align_face(img, face['landmarks'])
    
    # Get face region
    bbox = face['bbox'].astype(int)
    face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Preprocess
    face_img, _ = preprocess_image(face_img)
    
    # Get embedding
    embedding = face['embedding']
    return embedding / np.linalg.norm(embedding)

def aggregate_temporal_embeddings(embeddings: List[np.ndarray], 
                                qualities: List[float]) -> np.ndarray:
    """
    Aggregate embeddings with quality-weighted temporal consistency.
    """
    if not embeddings:
        return None
        
    # Quality-weighted average
    weighted_embeddings = [emb * q for emb, q in zip(embeddings, qualities)]
    avg_embedding = np.sum(weighted_embeddings, axis=0) / sum(qualities)
    
    return avg_embedding / np.linalg.norm(avg_embedding)

@app.route('/register_student', methods=['POST'])
def register_student():
    """
    Register a new student with their face embedding.
    """
    logger.info("Attempting to register student")
    try:
        # Validate input
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
            
        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
            
        student_id = request.form.get('student_id')
        if not validate_student_id(student_id):
            return jsonify({"error": "Invalid student ID"}), 400

        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check image quality
        quality_check, quality_error = check_image_quality(img)
        if not quality_check:
            return jsonify({"error": quality_error}), 400

        # Process faces
        face, error = process_multiple_faces(img)
        if error:
            return jsonify({"error": error}), 400

        # Extract embedding
        embedding = face['embedding']

        # Save data
        img_path = os.path.join(config['UPLOAD_FOLDER'], f'{student_id}.jpg')
        embedding_path = os.path.join(config['EMBEDDING_FOLDER'], f'{student_id}.json')
        
        cv2.imwrite(img_path, img)
        with open(embedding_path, 'w') as f:
            json.dump({
                'embedding': embedding.tolist(),
                'timestamp': datetime.now().isoformat(),
                'filename': file.filename
            }, f)

        logger.info(f"Successfully registered student {student_id}")
        return jsonify({
            "message": "Student registered successfully",
            "student_id": student_id
        }), 200

    except Exception as e:
        logger.error(f"Error registering student: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/register_student_multi_angle', methods=['POST'])
def register_student_multi_angle():
    """
    Enhanced multi-angle student registration with quality checks.
    """
    logger.info("Attempting to register student with multiple angles")
    try:
        # Validate student ID
        student_id = request.form.get('student_id')
        if not validate_student_id(student_id):
            return jsonify({"error": "Invalid student ID"}), 400

        # Validate all required angles are present
        embeddings = {}
        image_paths = {}
        quality_scores = {}
        
        for angle in config['REQUIRED_ANGLES']:
            image_key = f'image_{angle}'
            if image_key not in request.files:
                return jsonify({"error": f"Missing {angle} angle image"}), 400
                
            file = request.files[image_key]
            if not file or not allowed_file(file.filename):
                return jsonify({"error": f"Invalid file for {angle} angle"}), 400

            # Process image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Check image quality
            quality_check, quality_error = check_image_quality(img)
            if not quality_check:
                return jsonify({"error": f"{angle} angle: {quality_error}"}), 400

            # Process face
            face, error = process_multiple_faces(img)
            if error:
                return jsonify({"error": f"{angle} angle: {error}"}), 400

            # Validate face angle
            if not validate_face_angle(face, angle):
                return jsonify({"error": f"Incorrect face angle for {angle} view"}), 400

            # Add enhanced processing
            img, preprocess_info = preprocess_image(img)
            faces = enhance_face_detection(img)
            
            if not faces:
                return jsonify({"error": f"No valid face detected in {angle} angle"}), 400
                
            best_face = max(faces, key=lambda x: x['quality_score'])
            if best_face['quality_score'] < config['FACE_QUALITY']['MIN_QUALITY_SCORE']:
                return jsonify({"error": f"Face quality too low in {angle} angle"}), 400
                
            # Get enhanced embedding
            embedding = get_enhanced_embedding(img, best_face)
            
            embeddings[angle] = embedding
            quality_scores[angle] = best_face['quality_score']
            image_paths[angle] = os.path.join(config['UPLOAD_FOLDER'], f'{student_id}_{angle}.jpg')
            cv2.imwrite(image_paths[angle], img)

        # Quality-weighted embedding combination
        weights = np.array(list(quality_scores.values()))
        weights = weights / np.sum(weights)
        combined_embedding = np.sum([emb * w for emb, w in zip(embeddings.values(), weights)], axis=0)
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        # Save combined data
        embedding_path = os.path.join(config['EMBEDDING_FOLDER'], f'{student_id}.json')
        with open(embedding_path, 'w') as f:
            json.dump({
                'embedding': combined_embedding.tolist(),
                'individual_embeddings': {
                    angle: emb.tolist() for angle, emb in embeddings.items()
                },
                'image_paths': image_paths,
                'timestamp': datetime.now().isoformat()
            }, f)

        logger.info(f"Successfully registered student {student_id} with multiple angles")
        return jsonify({
            "message": "Student registered successfully with multiple angles",
            "student_id": student_id,
            "angles_processed": list(embeddings.keys())
        }), 200

    except Exception as e:
        logger.error(f"Error in enhanced multi-angle registration: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/recognize_student', methods=['POST'])
def recognize_student():
    """
    Recognize a student from their face.
    """
    logger.info("Attempting to recognize student")
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check image quality
        quality_check, quality_error = check_image_quality(img)
        if not quality_check:
            return jsonify({"error": quality_error}), 400

        # Process faces
        face, error = process_multiple_faces(img)
        if error:
            return jsonify({"error": error}), 400

        # Extract embedding
        embedding = face['embedding']

        # Compare with stored embeddings
        best_match = None
        highest_similarity = 0
        
        for student_file in os.listdir(config['EMBEDDING_FOLDER']):
            student_id = student_file.split('.')[0]
            stored_embedding = get_stored_embedding(student_id)
            
            if stored_embedding is not None:
                similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = student_id

        if highest_similarity > config['SIMILARITY_THRESHOLD']:
            logger.info(f"Student {best_match} recognized with similarity {highest_similarity}")
            return jsonify({
                "message": "Student recognized",
                "student_id": best_match,
                "similarity": float(highest_similarity)
            }), 200
        else:
            logger.info("No matching student found")
            return jsonify({"message": "No matching student found"}), 400

    except Exception as e:
        logger.error(f"Error recognizing student: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/batch_register', methods=['POST'])
def batch_register():
    """
    Register multiple students at once.
    """
    logger.info("Attempting batch registration")
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No images uploaded"}), 400
            
        results = []
        files = request.files.getlist('images')
        
        for file in files:
            try:
                if not allowed_file(file.filename):
                    results.append({
                        "status": "error",
                        "filename": file.filename,
                        "error": "Invalid file type"
                    })
                    continue

                # Extract student ID from filename (assuming format: studentid.jpg)
                student_id = file.filename.split('.')[0]
                if not validate_student_id(student_id):
                    results.append({
                        "status": "error",
                        "filename": file.filename,
                        "error": "Invalid student ID format"
                    })
                    continue

                # Process image (reusing single registration logic)
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                quality_check, quality_error = check_image_quality(img)
                if not quality_check:
                    results.append({
                        "status": "error",
                        "filename": file.filename,
                        "error": quality_error
                    })
                    continue

                face, error = process_multiple_faces(img)
                if error:
                    results.append({
                        "status": "error",
                        "filename": file.filename,
                        "error": error
                    })
                    continue

                # Process and save embedding
                embedding = face['embedding']

                img_path = os.path.join(config['UPLOAD_FOLDER'], f'{student_id}.jpg')
                embedding_path = os.path.join(config['EMBEDDING_FOLDER'], f'{student_id}.json')
                
                cv2.imwrite(img_path, img)
                with open(embedding_path, 'w') as f:
                    json.dump({
                        'embedding': embedding.tolist(),
                        'timestamp': datetime.now().isoformat(),
                        'filename': file.filename
                    }, f)

                results.append({
                    "status": "success",
                    "filename": file.filename,
                    "student_id": student_id
                })

            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": file.filename,
                    "error": str(e)
                })

        logger.info(f"Batch registration completed. Total processed: {len(results)}")
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Error in batch registration: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/recognize_group', methods=['POST'])
def recognize_group():
    """
    Recognize multiple students from a group photo.
    """
    logger.info("Attempting to recognize students from group photo")
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check image quality
        quality_check, quality_error = check_image_quality(img)
        if not quality_check:
            return jsonify({"error": quality_error}), 400

        # Detect all faces in the image
        faces = face_analyzer.get(img)
        
        if not faces:
            return jsonify({"error": "No faces detected in the image"}), 400

        # Process each face
        recognized_students = []
        unrecognized_faces = 0

        for face_idx, face in enumerate(faces):
            embedding = face['embedding']
            best_match = None
            highest_similarity = 0
            
            # Get face bounding box
            bbox = face['bbox'].astype(int)
            face_location = {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "width": int(bbox[2] - bbox[0]),
                "height": int(bbox[3] - bbox[1])
            }

            # Compare with stored embeddings
            for student_file in os.listdir(config['EMBEDDING_FOLDER']):
                student_id = student_file.split('.')[0]
                stored_embedding = get_stored_embedding(student_id)
                
                if stored_embedding is not None:
                    similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = student_id

            if highest_similarity > config['SIMILARITY_THRESHOLD']:
                recognized_students.append({
                    "student_id": best_match,
                    "similarity": float(highest_similarity),
                    "face_location": face_location,
                    "face_index": face_idx
                })
            else:
                unrecognized_faces += 1

        # Prepare response
        response = {
            "total_faces": len(faces),
            "recognized_count": len(recognized_students),
            "unrecognized_count": unrecognized_faces,
            "recognized_students": recognized_students
        }

        # Log the results
        logger.info(f"Group recognition completed. "
                   f"Total faces: {len(faces)}, "
                   f"Recognized: {len(recognized_students)}, "
                   f"Unrecognized: {unrecognized_faces}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in group recognition: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def extract_frames(video_path: str, fps: int = 2) -> Generator[np.ndarray, None, None]:
    """
    Extract frames from video at specified FPS.
    """
    video = cv2.VideoCapture(video_path)
    frame_interval = int(video.get(cv2.CAP_PROP_FPS) / fps)
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            yield frame
        frame_count += 1

    video.release()

def process_frame(frame: np.ndarray, frame_index: int) -> Dict:
    """
    Process a single frame to detect and recognize faces.
    """
    faces = face_analyzer.get(frame)
    frame_results = []
    
    for face in faces:
        embedding = face['embedding']
        bbox = face['bbox'].astype(int)
        
        # Find best match
        best_match = None
        highest_similarity = 0
        
        for student_file in os.listdir(config['EMBEDDING_FOLDER']):
            student_id = student_file.split('.')[0]
            stored_embedding = get_stored_embedding(student_id)
            
            if stored_embedding is not None:
                similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = student_id
        
        if highest_similarity > config['SIMILARITY_THRESHOLD']:
            frame_results.append({
                'student_id': best_match,
                'similarity': float(highest_similarity),
                'bbox': bbox.tolist(),
                'frame_index': frame_index
            })
    
    return {
        'frame_index': frame_index,
        'detections': frame_results
    }

def check_face_quality(face: dict) -> Tuple[bool, Optional[str]]:
    """
    Enhanced face quality assessment.
    """
    # Check face detection score
    if face.get('det_score', 0) < config['MIN_FACE_SCORE']:
        return False, "Low confidence face detection"

    # Check face size
    bbox = face['bbox']
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    if min(face_width, face_height) < config['MIN_FACE_SIZE']:
        return False, "Face too small"

    # Check face angle
    try:
        pitch, yaw, roll = face.get('pose', [0, 0, 0])
        if abs(yaw) > config['FACE_ANGLE_THRESHOLD'] or \
           abs(pitch) > config['FACE_ANGLE_THRESHOLD']:
            return False, "Face angle too large"
    except:
        pass

    return True, None

def calculate_weighted_similarity(embedding1: np.ndarray, embedding2: np.ndarray, quality_score: float) -> float:
    """
    Calculate similarity score weighted by face quality.
    """
    base_similarity = float(cosine_similarity([embedding1], [embedding2])[0][0])
    return base_similarity * quality_score

def process_frame_enhanced(frame: np.ndarray, frame_index: int) -> Dict:
    """
    Enhanced frame processing with quality assessment.
    """
    faces = face_analyzer.get(frame)
    frame_results = []
    
    for face in faces:
        # Check face quality
        quality_check, quality_error = check_face_quality(face)
        if not quality_check:
            continue

        embedding = face['embedding']
        bbox = face['bbox'].astype(int)
        quality_score = face.get('det_score', 0.9)
        
        # Find best match with weighted similarity
        best_match = None
        highest_similarity = 0
        
        for student_file in os.listdir(config['EMBEDDING_FOLDER']):
            student_id = student_file.split('.')[0]
            stored_embedding = get_stored_embedding(student_id)
            
            if stored_embedding is not None:
                similarity = calculate_weighted_similarity(embedding, stored_embedding, quality_score)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = student_id
        
        if highest_similarity > config['SIMILARITY_THRESHOLD']:
            frame_results.append({
                'student_id': best_match,
                'similarity': highest_similarity,
                'bbox': bbox.tolist(),
                'frame_index': frame_index,
                'quality_score': quality_score
            })
    
    return {
        'frame_index': frame_index,
        'detections': frame_results
    }

@app.route('/recognize_video', methods=['POST'])
def recognize_video():
    """
    Enhanced video recognition with temporal consistency and quality assessment.
    """
    logger.info("Starting enhanced video recognition")
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded"}), 400
            
        video_file = request.files['video']
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return jsonify({"error": "Invalid video format"}), 400

        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name

        # Process video frames
        all_results = []
        student_tracks = {}  # For temporal tracking
        frame_count = 0
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel frame processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for frame in extract_frames(video_path, fps=2):  # Process 2 frames per second
                futures.append(executor.submit(process_frame_enhanced, frame, frame_count))
                frame_count += 1
                
            # Collect results
            for future in futures:
                result = future.result()
                all_results.append(result)

        # Apply temporal consistency
        student_appearances = {}
        
        for frame_result in all_results:
            for detection in frame_result['detections']:
                student_id = detection['student_id']
                frame_idx = detection['frame_index']
                
                if student_id not in student_tracks:
                    student_tracks[student_id] = []
                
                student_tracks[student_id].append({
                    'frame_index': frame_idx,
                    'similarity': detection['similarity'],
                    'quality_score': detection['quality_score']
                })

        # Filter reliable detections
        for student_id, tracks in student_tracks.items():
            # Sort tracks by frame index
            tracks.sort(key=lambda x: x['frame_index'])
            
            # Check for consistent appearances
            reliable_tracks = []
            window_tracks = []
            
            for track in tracks:
                window_tracks.append(track)
                if len(window_tracks) > config['TEMPORAL_WINDOW']:
                    window_tracks.pop(0)
                
                if len(window_tracks) >= config['MIN_RECOGNITION_OCCURRENCES']:
                    avg_similarity = sum(t['similarity'] for t in window_tracks) / len(window_tracks)
                    if avg_similarity > config['SIMILARITY_THRESHOLD']:
                        reliable_tracks.append(track)
            
            if reliable_tracks:
                student_appearances[student_id] = {
                    'appearances': reliable_tracks,
                    'total_frames': len(reliable_tracks),
                    'average_similarity': sum(t['similarity'] for t in reliable_tracks) / len(reliable_tracks),
                    'confidence_score': len(reliable_tracks) / frame_count
                }

        # Clean up temporary file
        os.unlink(video_path)
        
        processing_time = time.time() - start_time
        
        response = {
            'total_frames_processed': frame_count,
            'processing_time_seconds': processing_time,
            'recognized_students': student_appearances
        }
        
        logger.info(f"Video processing completed. Processed {frame_count} frames in {processing_time:.2f} seconds")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in enhanced video recognition: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


def validate_video(video_file) -> Tuple[bool, Optional[str]]:
    """
    Validate video file size and length.
    """
    try:
        # Check file size
        video_file.seek(0, 2)  # Seek to end
        file_size = video_file.tell()
        video_file.seek(0)  # Reset position
        
        if file_size > config['MAX_VIDEO_SIZE_MB'] * 1024 * 1024:
            return False, f"Video size exceeds {config['MAX_VIDEO_SIZE_MB']}MB limit"
            
        # Create temporary file to check video length
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            cap = cv2.VideoCapture(temp_video.name)
            
            if not cap.isOpened():
                return False, "Unable to open video file"
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps
            
            if duration > config['MAX_VIDEO_LENGTH_SECONDS']:
                return False, f"Video length exceeds {config['MAX_VIDEO_LENGTH_SECONDS']} seconds limit"
                
            cap.release()
            
        return True, None
        
    except Exception as e:
        return False, f"Error validating video: {str(e)}"



@app.errorhandler(Exception)
def handle_error(error):
    """
    Global error handler
    """
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6969 )