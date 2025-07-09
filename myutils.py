import torch
from torchvision import transforms
from PIL import Image
import joblib
import timm
import warnings
import cv2 as cv
from PIL import Image
import os
import sys
import contextlib
from datetime import datetime
from megadetector.detection import run_detector
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

MODEL_PATH = "model/model.joblib"
CONFIDENCE_THRESHOLD_EARLY_STOP = 0.80

frame_interval = 5  # Analyze every 5th frame
# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# Define the label map
label_map = {
    'badger': 0, 'bird': 1, 'boar': 2, 'butterfly': 3,
    'cat': 4, 'dog': 5, 'fox': 6, 'lizard': 7,
    'podolic_cow': 8, 'porcupine': 9, 'weasel': 10,
    'wolf': 11, 'other': 12
}

def model_setup():
    """ Load the feature extractor model and the classifier.
    :return: Tuple containing the feature extractor model, the classifier, and the device.
    """
    with suppress_output():
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        detection_model = run_detector.load_detector("MDV5A", force_cpu=(device == "cpu"))
        
        # Load feature extractor model (ViT without classifier)
        model_feat = timm.create_model('vit_base_patch16_224', pretrained=True)
        model_feat.reset_classifier(0)  # Remove classification head
        model_feat.eval()
        model_feat = model_feat.to(device)
        
        # Load the Logistic Regression classifier
        classifier = joblib.load(MODEL_PATH)
    
    return model_feat, classifier, device, detection_model

def write_csv(video_path, label, confidence, csv_file="predictions.csv"):
    """
    Write the predicted label to a CSV file.
    :param video_path: Path to the video file
    :param label: Predicted label
    :param confidence: Confidence score of the prediction
    :param csv_file: Name of the CSV file to write to
    :return: None
    """
    if label is not None:
        base, ext = os.path.splitext(video_path)
        folder = os.path.dirname(base)
        csv_path = f"{folder}/{csv_file}.csv"
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write("video_path,label,confidence,timestamp\n")  # Write header
        with open(csv_path, 'a') as f:
            f.write(f"\"{video_path}\",{label},{confidence:.5f},{timestamp_str}\n")  # Append prediction

def rename_video(video_path, label):
    """
    Rename the video file based on the predicted label.
    If a file with the new name already exists, add a counter to avoid overwriting.
    :param video_path: Path to the video file
    :param label: Predicted label
    :return: None
    """
    base, ext = os.path.splitext(video_path)
    folder = os.path.dirname(base)
    new_name = os.path.join(folder, f"{label}{ext}")
    counter = 1

    # Increment the filename if it already exists
    while os.path.exists(new_name):
        new_name = os.path.join(folder, f"{label}_{counter+1}{ext}")
        counter += 1

    os.rename(video_path, new_name)

    return new_name

def get_best_frame(video, detection_model, conf_threshold=0.30, frame_interval=5):
    """ Get the best frame from the video based on detection confidence.
    :param video: OpenCV VideoCapture object
    :param device: Device to run the model on ('cpu' or 'cuda')
    :param detection_model: Model for object detection
    :param conf_threshold: Confidence threshold for early stopping
    :param frame_interval: Interval to skip frames (e.g., analyze every 5th frame)
    :return: Best frame and its bounding box
    """
    best_conf = 0.0
    best_frame = None
    frame_count = 0
    best_bounding_box = None
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  # Skip frames

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Apply detection
        result = detection_model.generate_detections_one_image(image)
        detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]

        # Choose the best detection
        if detections_above_threshold:
            max_conf = max(d['conf'] for d in detections_above_threshold)
            if max_conf > best_conf:
                best_conf = max_conf
                best_frame = frame.copy()
                best_bounding_box = [d['bbox'] for d in detections_above_threshold if d['conf'] == max_conf][0]

                if best_conf >= conf_threshold:
                    break 

    video.release()
    return best_frame, best_bounding_box

def crop_bounding_box(frame, bounding_box):
    """ Crop the image based on the bounding box.
    :param frame: PIL Image object
    :param bounding_box: List or tuple with (x_min, y_min, width, height)
    :return: Cropped PIL Image
    """
    img_width, img_height = frame.size
    x_min, y_min, width, height = bounding_box
    # Coordinates for cropping
    left = int(x_min * img_width)
    top = int(y_min * img_height)
    right = int((x_min + width) * img_width)
    bottom = int((y_min + height) * img_height)
    
    # Crop the image
    cropped = frame.crop((left, top, right, bottom))
    return cropped

def classify_single_video(video_path, model_feat, classifier, detection_model, device, frame_interval=5):
    video = cv.VideoCapture(video_path)
    best_frame = None
    best_bounding_box = []
    try:
        best_frame, best_bounding_box = get_best_frame(video, detection_model, 0.30, frame_interval)

        inv_label_map = {v: k for k, v in label_map.items()} 

        # Load the image transformation
        image = crop_bounding_box(Image.fromarray(cv.cvtColor(best_frame, cv.COLOR_BGR2RGB)).convert("RGB"), best_bounding_box)

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = model_feat(input_tensor)
            features_np = features.cpu().numpy() 

        # Predict using the classifier
        pred = classifier.predict(features_np)[0]
        pred_label = inv_label_map.get(pred, "Unknown")
    
        probs = classifier.predict_proba(features_np)[0]

        # Calculate confidence
        confidence = probs[pred]
    except Exception as e:
        pred_label = None
        confidence = None
    return pred_label, confidence


def classify_multiple_videos(video_folder, model_feat, classifier, detection_model, device, frame_interval=5):
    results = []
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, filename)
            best_label, best_conf = classify_single_video(video_path, model_feat, classifier, detection_model, device, frame_interval)
            results.append((video_path, best_label, best_conf))
    return results