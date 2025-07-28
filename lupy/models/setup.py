import os
import timm
import torch
import joblib
from lupy.utils.cli_controller import suppress_output
from megadetector.detection import run_detector

def models_setup():
    with suppress_output():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        detection_model = run_detector.load_detector("MDV5A", force_cpu=(device == "cpu"))
        
        model_feat = timm.create_model('vit_base_patch16_224', pretrained=True)
        model_feat.reset_classifier(0)
        model_feat.eval().to(device)

        os.environ['TESSDATA_PREFIX'] = os.path.join(os.path.dirname(__file__), "..", "models", "tessdata")
        classifier = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "classification_model.joblib"))

        return model_feat, classifier, device, detection_model
