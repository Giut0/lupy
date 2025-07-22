import os
import torch
import cv2 as cv
from PIL import Image
from lupy.utils.constants import label_map, transform
from lupy.utils.datetime_extraction import extract_datetime, format_datetime
from lupy.utils.video_processing import crop_bounding_box, draw_bbox, get_best_frame

def classify_single_video(video_path, model_feat, classifier, detection_model, device, save_datetime=False, img_save=False, frame_interval=5):
    """ Classify a single video and return the best label and confidence.
    :param video_path: Path to the video file
    :param model_feat: Feature extractor model
    :param classifier: Classifier model
    :param detection_model: Object detection model
    :param device: Device to run the model on ('cpu' or 'cuda')
    :param save_datetime: Flag to save date and time from the video
    :param img_save: Directory to save annotated images, or False to skip saving
    :param frame_interval: Interval to skip frames (e.g., analyze every 5th frame)
    :return: Tuple with best label, confidence, and formatted date/time string
    """
    video = cv.VideoCapture(video_path)
    best_frame = None
    best_bounding_box = []
    formatted_datetime = None
    try:
        best_frame, best_bounding_box = get_best_frame(video, detection_model, 0.30, frame_interval)

        if save_datetime:
            date, time = extract_datetime(best_frame)
            if date and time:
                formatted_datetime = format_datetime(date, time)

        inv_label_map = {v: k for k, v in label_map.items()}


        best_img = Image.fromarray(cv.cvtColor(best_frame, cv.COLOR_BGR2RGB))
        # Load the image transformation
        image = crop_bounding_box(best_img, best_bounding_box)

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
        # Annotated img saving
        if img_save and best_frame is not None:

            if not os.path.exists(img_save):
                os.makedirs(img_save)
        
            base_name = os.path.splitext(os.path.basename(video_path))[0]

            img_path = os.path.join(img_save, base_name + ".png")

            annotate_frame = draw_bbox(best_img, best_bounding_box, pred_label, confidence)
            annotate_frame.save(img_path, compress_level=0)

    except Exception as e:
        pred_label = None
        confidence = None

    return pred_label, confidence, formatted_datetime

def classify_multiple_videos(video_folder, model_feat, classifier, detection_model, device, save_datetime=False, img_save=False, frame_interval=5):
    """ Classify multiple videos in a folder and return results.
    :param video_folder: Path to the folder containing video files
    :param model_feat: Feature extractor model
    :param classifier: Classifier model
    :param detection_model: Object detection model
    :param device: Device to run the model on ('cpu' or 'cuda')
    :param save_datetime: Flag to save date and time from the video
    :param img_save: Directory to save annotated images, or False to skip saving
    :param frame_interval: Interval to skip frames (e.g., analyze every 5th frame)
    :return: List of tuples with video path, best label, confidence, and formatted date/time string
    """
    results = []
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, filename)
            best_label, best_conf, formatted_datetime = classify_single_video(video_path, model_feat, classifier, detection_model, device, save_datetime, img_save, frame_interval)
            results.append((video_path, best_label, best_conf, formatted_datetime))
    return results
