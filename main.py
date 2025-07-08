#!/usr/bin/env python3
import typer
import myutils

def main(
    #CLI options 
    video_path: str = typer.Option(None, "--path", "-p" ,help= "Video path" ),
    version: bool = typer.Option(False, "--version", "-v", help="Get the current version"),
    rename: bool = typer.Option(False, "--rename", "-r", help="Rename the video file"),
    write_csv: str = typer.Option(None, "--csv", "-c" ,help= "Write csv file")):

    #CLI --help documentation 
    '''
    Random password generator
    '''

    model_feat, clf, device = myutils.model_setup()

    if version:
        typer.echo("Current version: 1.7v")
        raise typer.Exit()
    
    if video_path:
        best_label, best_conf  = myutils.classificate_single_video(video_path, model_feat, clf, device)
        if rename:
            myutils.rename_video(video_path, best_label)
        
        if write_csv:
            myutils.write_csv(video_path, best_label, confidence=best_conf, csv_file=write_csv)
            
        print(f"Best frame label: {best_label} (Confidence: {best_conf:.2f})")

if __name__ == '__main__':
    typer.run(main)