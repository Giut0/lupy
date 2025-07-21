#!/usr/bin/env python3
import os
import typer
from lupy import cli_controller
from lupy.models.setup import models_setup


def main(
    video_path: str = typer.Option(None, "--path", "-p", help="📹  Path to a single video"),
    video_folder: str = typer.Option(None, "--folder", "-f", help="📁  Path to folder with multiple videos"),
    version: bool = typer.Option(False, "--version", "-v", help="🔢  Show current version"),
    rename: bool = typer.Option(False, "--rename", "-r", help="✏️  Rename video(s) using prediction"),
    write_csv: str = typer.Option(None, "--csv", "-c", help="📄  Save results to a CSV file"),
    save_datetime: bool = typer.Option(False, "--time", "-t", help="🗓️  Save extracted date and time from video"),
    img_save: str = typer.Option(None, "--img-save", "-i", help="🖼️  Save images from video frames"),
):
    """
    🐾 Lupy - Camera Trap Video Classification Tool
    """
    typer.echo("\n🚀 Starting Lupy...\n")

    if version:
        typer.echo("📦 Current version: 1.3v\n")
        raise typer.Exit(code=0)

    # Model setup
    try:
        model_feat, classifier, device, detection_model = models_setup()
        typer.echo("✅ Model setup complete.\n")
    except Exception as e:
        typer.echo(f"⛔ Error during model setup: {e}\n")
        raise typer.Exit(code=1)

    # Single video processing
    if video_path:
        if not os.path.isfile(video_path):
            typer.echo(f"⛔ Video file not found: {video_path}\n")
            raise typer.Exit(code=1)
        cli_controller.handle_single_video(video_path, model_feat, classifier, detection_model, device, rename, write_csv, save_datetime, img_save)

    # Multiple videos processing
    elif video_folder:
        if not os.path.isdir(video_folder):
            typer.echo(f"⛔ Video folder not found: {video_folder}\n")
            raise typer.Exit(code=1)
        cli_controller.handle_multiple_videos(video_folder, model_feat, classifier, detection_model, device, rename, write_csv, save_datetime, img_save)

    else:
        typer.echo("⛔ No video file or folder specified. Use --path or --folder.\n")
        raise typer.Exit(code=1)

    typer.echo("\n🎉 Lupy processing complete! Thank you for using Lupy!\n")
    


def lupy():
    typer.run(main)

if __name__ == '__main__':
    typer.run(main)
