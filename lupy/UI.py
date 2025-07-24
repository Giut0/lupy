
import os
import zipfile
import streamlit as st
from lupy.utils.file_ops import save_csv
from lupy.models.setup import models_setup
from lupy.models.video_classification import classify_single_video
import logging
logging.getLogger("timm").setLevel(logging.ERROR)

# Model setup
model_feat, classifier, device, detection_model = models_setup()

img_dir = "annotated_frames"
video_dir = "uploaded_videos"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

st.title("🐾 Lupy - Camera Trap Video Classification Tool")

if "results" not in st.session_state:
    st.session_state.results = []
if "single_video_bytes" not in st.session_state:
    st.session_state.single_video_bytes = None

# Upload video
video_files = st.file_uploader("Upload one or more videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

# Single Video Player
if video_files and len(video_files) == 1:
    video_file = video_files[0]
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.session_state.single_video_bytes = video_bytes

# Clear cache buttons
st.markdown("---")
if st.button("🧹 Clear cache (uploaded videos + annotated frames)"):
    try:
        for f in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, f))
        os.rmdir(video_dir)
    except Exception:
        pass

    try:
        for f in os.listdir(img_dir):
            os.remove(os.path.join(img_dir, f))
    except Exception:
        pass

    st.success("🧼 Cache cleared!")

# Video classification
if video_files:
    st.write(f"📦 {len(video_files)} video(s) uploaded.")
    if st.button("Run analysis on all videos"):
        with st.spinner("Processing videos..."):
            st.session_state.results = []

            # Old temporary files cleanup
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

                video_name = os.path.basename(video_file.name)
                video_path = os.path.join(video_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)

                with open(video_path, "wb") as f:
                    f.write(video_bytes)

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

# Mostra risultati
if len(st.session_state.results) > 0:
    # Show results 
    st.markdown("---")
    st.header("📊 Summary of Results")
    for name, label, conf, dt, _ in st.session_state.results:
        if label is None:
            continue
        st.write(f'• Video: `{name}` → Label: `{label}` (`{conf:.2f}` conf.) - 📅 Timestamp: `{dt or "n/a"}`')

    if len(st.session_state.results) > 1:
        # Export results
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

        # Annotated img ZIP Download 
        zip_path = os.path.join(img_dir, "annotated_images.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(img_dir):
                if file.endswith(".png"):
                    file_path = os.path.join(img_dir, file)
                    zipf.write(file_path, arcname=file)

        with open(zip_path, "rb") as f:
            st.download_button(
                label="🖼️ Download annotated images ZIP",
                data=f,
                file_name="annotated_images.zip",
                mime="application/zip"
            )

else:
    if video_files:
        st.info("➡️ Click 'Run analysis on all videos' to generate results.")

# Exit button
st.markdown("---")
if st.button("🛑 Exit application"):
    st.info("⛔ Exiting the application...")
    os._exit(0)