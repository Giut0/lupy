import cv2 as cv
import numpy as np
from PIL import Image

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

def draw_bbox(image, bbox, label, confidence):
    """ Draw bounding box and label on the image.
    :param image: PIL Image object
    :param bbox: Bounding box coordinates (x_min, y_min, width, height)
    :param label: Predicted label
    :param confidence: Confidence score
    :return: Annotated PIL Image
    """
    img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    h, w, _ = img.shape
    x_min, y_min, box_w, box_h = bbox
    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int((x_min + box_w) * w)
    y2 = int((y_min + box_h) * h)

    text = f"{label} ({confidence:.2f})"
    color = (0, 0, 255)  # Red

    # Draw box
    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # Put label
    cv.putText(img, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
