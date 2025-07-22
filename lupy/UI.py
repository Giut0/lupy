from lupy.models.setup import models_setup
import streamlit as st
import tempfile
from lupy.models.video_classification import classify_single_video
import os
from datetime import datetime
import zipfile

model_feat, classifier, device, detection_model = models_setup()

def save_csv(results=None, csv_file="predictions.csv"):
    if results is None:
        return
    with open(csv_file, 'w') as f:
        f.write("video_path,label,confidence,video_timestamp,detection_timestamp\n")
        for name, label, conf, dt, _ in results:
            if label is None:
                continue
            detection_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\"{name}\",{label},{conf:.5f},{dt},{detection_timestamp}\n")

st.title("🐾 Lupy - Camera Trap Video Classification Tool")

# Session state for results and bytes
if "results" not in st.session_state:
    st.session_state.results = []
if "single_video_bytes" not in st.session_state:
    st.session_state.single_video_bytes = None

# File upload
video_files = st.file_uploader("Upload one or more videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

# Directory to save annotated images
img_dir = "annotated_frames"
os.makedirs(img_dir, exist_ok=True)

# Show player if single video is uploaded
if video_files and len(video_files) == 1:
    video_file = video_files[0]
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.session_state.single_video_bytes = video_bytes

# Run analysis
if video_files:
    st.write(f"📦 {len(video_files)} video(s) uploaded.")
    if st.button("Run analysis on all videos"):
        with st.spinner("Processing videos..."):
            st.session_state.results = []

            # Clean up old temporary images
            for file in os.listdir(img_dir):
                if "tmp" in file.lower():
                    try:
                        os.remove(os.path.join(img_dir, file))
                    except Exception:
                        pass

            for idx, video_file in enumerate(video_files):
                if len(video_files) == 1:
                    video_bytes = st.session_state.single_video_bytes
                else:
                    video_bytes = video_file.read()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(video_bytes)
                    video_path = tfile.name

                try:
                    label, conf, dt = classify_single_video(
                        video_path=video_path,
                        model_feat=model_feat,
                        classifier=classifier,
                        detection_model=detection_model,
                        device=device,
                        save_datetime=True,
                        img_save=img_dir,
                        frame_interval=5
                    )
                    if label is not None:
                        st.session_state.results.append((video_file.name, label, conf, dt, video_path))
                except Exception:
                    st.session_state.results.append((video_file.name, "Error", None, None, None))

        st.success("✅ Analysis completed!")

# Show results if available
if len(st.session_state.results) > 1:
    st.markdown("---")
    st.header("📄 Textual Results")
    for name, label, conf, dt, _ in st.session_state.results:
        if label is None:
            continue
        st.write(f'• Video: `{name}` → Label: `{label}` (`{conf:.2f}` conf.) - 📅 Timestamp: `{dt or "n/a"}`')


    # Export section
    st.markdown("---")
    st.subheader("📤 Export Results")

    # CSV Download
    csv_path = "predictions.csv"
    save_csv(st.session_state.results, csv_path)
    with open(csv_path, "rb") as f:
        st.download_button(
            label="📄 Download predictions.csv",
            data=f,
            file_name="predictions.csv",
            mime="text/csv"
        )

    # ZIP Download
    zip_path = os.path.join(img_dir, "annotated_images.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(img_dir):
            if file.endswith(".png") and "tmp" not in file.lower() and not file.startswith("._"):
                file_path = os.path.join(img_dir, file)
                zipf.write(file_path, arcname=file)

    with open(zip_path, "rb") as f:
        st.download_button(
            label="🖼️ Download annotated images ZIP", # TODO: fix image saving
            data=f,
            file_name="annotated_images.zip",
            mime="application/zip"
        )

else:
    # If videos uploaded but no results yet
    if video_files:
        st.info("➡️ Click 'Run analysis on all videos' to generate results.")
