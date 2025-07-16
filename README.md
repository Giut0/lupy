# 🐾 Lupy - AI-Powered Camera Trap Video Classification

**Lupy** is an AI-based command-line tool designed to process and classify wildlife videos from camera traps.  
It uses **computer vision** to automatically detect animals and assign class labels through a custom-trained model.

> Built for conservationists, ecologists, and AI enthusiasts working with large-scale wildlife video data.

---

## 🚀 Key Features

- 🧠 **AI-powered** wildlife detection and classification
- 🌍 Focused on species found in the **Alta Murgia National Park**, Italy
- 🎥 Process single video files or entire folders
- ✏️ Rename videos based on AI predictions
- 📄 Export classification results to CSV
- ⚙️ Combines **MegaDetector** with a **custom classifier**
- 🔍 Tesseract OCR engine using **pytesseract**

---

## 🦊 Species Classification

Lupy’s AI model has been trained specifically to classify wildlife species typical of the **Alta Murgia region**.  
It supports the following animal categories:

- `badger`, `bird`, `boar`, `butterfly`, `cat`, `dog`,  
- `fox`, `lizard`, `podolic_cow`, `porcupine`, `weasel`, `wolf`,  
- `other` (for unrecognized or less common species)

The model is based on real video data collected from camera traps in the Alta Murgia landscape.

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Giut0/lupy.git
cd lupy
```

### 2. Install with `pip`

```bash
pip install .
```

### 3. Install `tesseract`
### 🪟 Windows

1. Download the precompiled binary:

   👉 https://github.com/UB-Mannheim/tesseract/wiki

2. Install it (or unzip the portable `.zip` version).
### 🐧 Linux

#### Debian / Ubuntu

```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### Arch / Manjaro

```bash
sudo pacman -S tesseract
```


## 🧪 Usage

After installation, run `lupy` from your terminal.

### Show help

```bash
lupy --help
```

### Show version

```bash
lupy --version
```

### Classify a single video

```bash
lupy -p /path/to/video.mp4
```

### Classify a folder of videos

```bash
lupy -f /path/to/folder/
```

### Rename video(s) using predicted labels

```bash
lupy -p /path/to/video.mp4 --rename
# or
lupy -f /path/to/folder/ --rename
```

### Extract video timestamp
```bash
lupy -p /path/to/video.mp4 -t
# or
lupy -f /path/to/folder/ -t
```

### Export results to CSV

```bash
lupy -p /path/to/video.mp4 --csv results
```

You can also combine options:

```bash
lupy -f /path/to/folder/ --rename --csv results
```


## ⚙️ Requirements

- Python 3.8+
- Deep learning model files (MegaDetector + custom classifier)
- Python dependencies (installed via `pip install -r requirements.txt`)
- Tesseract OCR

## 🤖 AI Model & Training Details

The custom classifier used in Lupy is part of a broader **AI-powered computer vision pipeline** developed for wildlife monitoring.

To explore the training process and model architecture, check out the companion repository:

👉 **[Wildlife Computer Vision Model Repository](https://github.com/Giut0/Murgia-AI-Wildlife-Track)**


## 📄 License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** license.  
You may use, modify, and share the code for **non-commercial purposes only**, and you must give appropriate credit.
