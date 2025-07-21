import cv2
import re
import pytesseract
from datetime import datetime

def format_datetime(date_str, time_str):
    """Converts date/time from OCR into standard ISO format: '%Y-%m-%d %H:%M:%S'
    :param date_str: Date string from OCR
    :param time_str: Time string from OCR
    :return: Formatted date/time string or None if no format matches
    """
    # Try different common formats
    for date_fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
                     '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
                     '%Y/%m/%d', '%Y-%m-%d']:
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} %H:%M:%S")
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

    return None

def extract_datetime_pattern(ocr_text):
    """ Extract date and time from OCR text using regex patterns.
    :param ocr_text: Text extracted from OCR
    :return: Tuple with date and time strings or (None, None) if not found
    """
    ocr_text = re.sub(r'(\d{2}:\d{2})\s?:\s?(\d{2})', r'\1:\2', ocr_text)

    date_time_patterns = [
        r"\b(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<date>\d{2}[\/\-\.]\d{1,3}[\/\-\.]\d{2,4})\b",
        r"\b(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<date>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b",
        r"\b(?P<date>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<date>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<date>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
    ]

    for pattern in date_time_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            return match.group('date'), match.group('time')

    return None, None

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [
        gray,
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.bitwise_not(gray)
    ]

def extract_from_region(region):
    """ Apply OCR and search for date/time in the region """
    for proc in preprocess(region):
        text = pytesseract.image_to_string(proc, config="--psm 6", lang='eng+ita')

        text = pytesseract.image_to_string(proc, config="--psm 6")
        date, time = extract_datetime_pattern(text)

        if date and time:
            return date, time
    return None, None

def extract_datetime(img):
    """ Extract date and time from the image using OCR.
    :param img: Path to the image file or OpenCV image object
    :return: Tuple with date and time strings or (None, None) if not found
    """
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # Lower part (last strip of the image)
    lower = img[int(h * 0.80):, :]
    date, time = extract_from_region(lower)
    if date and time:
        return date, time

    # Higher part (first strip)
    upper = img[:int(h * 0.20), :]
    date, time = extract_from_region(upper)
    if date and time:
        return date, time

    return None, None