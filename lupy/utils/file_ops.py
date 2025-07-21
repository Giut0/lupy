import os
from datetime import datetime

def write_to_csv(video_path, label, confidence, formatted_datetime, csv_file="predictions.csv"):
    """
    Write the predicted label and metadata to a CSV file.

    Parameters:
        video_path (str): Path to the video file.
        label (str): Predicted label (must not be None).
        confidence (float): Confidence score of the prediction (must not be None).
        formatted_datetime (str): Formatted date and time string extracted from the video.
        csv_file (str): Name of the CSV file (without extension).

    Returns:
        None
    """
    if label is None or confidence is None:
        return 

    base, _ = os.path.splitext(video_path)
    folder = os.path.dirname(base)
    csv_path = os.path.join(folder, f"{csv_file}.csv")
    detection_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Write header
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("video_path,label,confidence,video_timestamp,detection_timestamp\n")

    # Append prediction data
    with open(csv_path, 'a') as f:
        f.write(f"\"{video_path}\",{label},{confidence:.5f},{formatted_datetime},{detection_timestamp}\n")


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
