import os
import sys
import typer
import contextlib
from lupy.utils.file_ops import rename_video, write_to_csv
from lupy.models.video_classification import classify_multiple_videos, classify_single_video

def handle_single_video(video_path, model_feat, classifier, detection_model, device, rename, write_csv, save_datetime, img_save):
    typer.echo(f"🔍 Processing single video: {video_path}")
    best_label, best_conf, formatted_datetime = classify_single_video(
        video_path, model_feat, classifier, detection_model, device, save_datetime, img_save
    )

    if best_label is None or best_conf is None:
        typer.echo("\n⚠️ No animal detected in the video.\n")
        raise typer.Exit(code=1)

    filename = os.path.basename(video_path)

    if rename:
        video_path = rename_video(video_path, best_label)

    typer.echo(f"  └ Video: {filename} -- Label: {best_label} (Confidence: {best_conf:.2f}) -- Timestamp: {formatted_datetime}")
    if img_save:
        typer.echo(f"\n💾 Annotated image saved: {img_save}\n")

    if write_csv:
        write_to_csv(
            video_path,
            best_label,
            confidence=best_conf,
            formatted_datetime=formatted_datetime,
            csv_file=write_csv
        )
        typer.echo(f"\n💾 Logged to CSV: {write_csv}\n")


def handle_multiple_videos(video_folder, model_feat, classifier, detection_model, device, rename, write_csv, save_datetime, img_save):
    typer.echo(f"📁 Processing all videos in folder: {video_folder}")
    results = classify_multiple_videos(
        video_folder, model_feat, classifier, detection_model, device, save_datetime, img_save
    )

    if not results:
        typer.echo("\n⛔ No animal videos found in the specified folder.\n")
        raise typer.Exit(code=1)

    for video_path, best_label, best_conf, formatted_datetime in results:
        if best_label is None or best_conf is None:
            typer.echo(f"  └ ⚠️ No animal detected in video: {video_path}, skipping...")
            continue

        filename = os.path.basename(video_path)

        if rename:
            video_path = rename_video(video_path, best_label)

        typer.echo(f"  └ Video: {filename} -- Label: {best_label} (Confidence: {best_conf:.2f}) -- Timestamp: {formatted_datetime}")

        if write_csv:
            write_to_csv(
                video_path,
                best_label,
                confidence=best_conf,
                formatted_datetime=formatted_datetime,
                csv_file=write_csv
            )

    if img_save:
        typer.echo(f"\n💾 All annotated images saved to: {img_save}\n")

    if not write_csv and typer.confirm("\n❓ Would you like to save the results to a CSV file?", default=False):
        csv_name = typer.prompt("\n✏️  Enter CSV filename (e.g., results.csv)")
        for video_path, best_label, best_conf, formatted_datetime in results:
            if best_label is not None:
                typer.echo(f"  └ Saving {video_path} with label '{best_label}' and confidence {best_conf:.2f} to {csv_name}")
                write_to_csv(
                    video_path,
                    best_label,
                    confidence=best_conf,
                    formatted_datetime=formatted_datetime,
                    csv_file=csv_name
                )
        typer.echo(f"\n💾 Logged all results to CSV: {video_folder}/{csv_name}.csv\n")

@contextlib.contextmanager
def suppress_output():
    """
    Context manager per sopprimere stdout e stderr temporaneamente.
    Utile per nascondere log di librerie esterne.
    """
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