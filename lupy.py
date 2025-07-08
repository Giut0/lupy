#!/usr/bin/env python3
import typer
import myutils
import os

def main(
    # CLI options
    video_path: str = typer.Option(None, "--path", "-p", help="📹  Path to a single video"),
    video_folder: str = typer.Option(None, "--folder", "-f", help="📁  Path to folder with multiple videos"),
    version: bool = typer.Option(False, "--version", "-v", help="🔢  Show current version"),
    rename: bool = typer.Option(False, "--rename", "-r", help="✏️  Rename video(s) using prediction"),
    write_csv: str = typer.Option(None, "--csv", "-c", help="📄  Save results to a CSV file")
):
    """
    🐾 Lupy - Camera Trap Video Classification Tool

    Use this tool to classify wildlife videos using MegaDetector and a custom classifier.
    """

    typer.echo("\n🚀 Starting Lupy...\n")

    try:
        model_feat, classifier, device, detection_model = myutils.model_setup()
        typer.echo("✅ Model setup complete.\n")
    except Exception as e:
        typer.echo(f"⛔ Error during model setup: {e}\n")
        raise typer.Exit(code=1)

    if version:
        typer.echo("📦 Current version: 1.0v\n")
        raise typer.Exit()

    # Check if the file or folder exists
    if video_path:
        if not os.path.isfile(video_path):
            typer.echo(f"⛔ Video file not found: {video_path}\n")
            raise typer.Exit(code=1)
    elif video_folder:
        if not os.path.isdir(video_folder):
            typer.echo(f"⛔ Video folder not found: {video_folder}\n")
            raise typer.Exit(code=1)
    else:
        typer.echo("⛔ No video file or folder specified. Use --path or --folder.\n")
        raise typer.Exit(code=1)

    if video_path:
        typer.echo(f"🔍 Processing single video: {video_path}")
        best_label, best_conf = myutils.classify_single_video(video_path, model_feat, classifier, detection_model, device)
        filename = os.path.basename(video_path)

        if rename:
            myutils.rename_video(video_path, best_label)

        typer.echo(f"  └ Video: {filename} -- Label: {best_label} (Confidence: {best_conf:.2f})")

        if write_csv:
            myutils.write_csv(video_path, best_label, confidence=best_conf, csv_file=write_csv)
            typer.echo(f"\n💾 Logged to CSV: {write_csv}")

    elif video_folder:
        typer.echo(f"📁 Processing all videos in folder: {video_folder}")

        results = myutils.classify_multiple_videos(video_folder, model_feat, classifier, detection_model, device)

        for result in results:
            video_path, best_label, best_conf = result
            filename = os.path.basename(video_path)

            if rename:
                myutils.rename_video(video_path, best_label)

            typer.echo(f"  └ Video: {filename} -- Label: {best_label} (Confidence: {best_conf:.2f})")

            if write_csv:
                myutils.write_csv(video_path, best_label, confidence=best_conf, csv_file=write_csv)

        if write_csv:
            typer.echo(f"\n💾 Logged all results to CSV: {write_csv}\n")
        else:
            if typer.confirm("\n❓ Would you like to save the results to a CSV file?", default=False):
                    csv_name = typer.prompt("\n✏️ Enter CSV filename (e.g., results.csv)")
                    for result in results:
                        video_path, best_label, best_conf = result
                        typer.echo(f"  └ Saving {video_path} with label \'{best_label}\' and confidence {best_conf:.2f} to {csv_name}")
                        myutils.write_csv(video_path, best_label, confidence=best_conf, csv_file=csv_name)
                    typer.echo(f"\n💾 Logged all results to CSV: {write_csv}\n")

    typer.echo("🎉 Lupy processing complete! Thank you for using Lupy!\n")
    
if __name__ == '__main__':
    typer.run(main)
