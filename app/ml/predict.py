import os
import re
import joblib
from typing import Optional
from app.ml.train import PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
ENCODER_DIR = os.path.join(PROJECT_ROOT, 'encoders')
MODEL_PATTERN = r'text_classifier_v(\d+)\.pkl'
ENCODER_PATTERN = r'label_encoder_v(\d+)\.pkl'

class Predictor:
    def __init__(self, model_version: Optional[int] = None, encoder_version: Optional[int] = None):
        self.model_path, self.encoder_path = self._get_model_and_encoder_paths(model_version, encoder_version)

        if not os.path.exists(self.model_path) or not os.path.exists(self.encoder_path):
            raise FileNotFoundError("Model or label encoder not found. Train the model first.")

        self.model = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
        print(f"Loaded model: {self.model_path}")
        print(f"Loaded label encoder: {self.encoder_path}")

    def _get_model_and_encoder_paths(self, model_version: Optional[int], encoder_version: Optional[int]) -> tuple[str, str]:
        def find_latest_versioned_file(pattern: str, prefix: str, dir: str) -> str:
            files = [f for f in os.listdir(dir) if re.match(pattern, f)]
            if not files:
                raise FileNotFoundError(f"No {prefix} files found in {dir}")

            files.sort(key=lambda x: int(re.search(pattern, x).group(1)), reverse=True)
            return os.path.join(dir, files[0])

        # Get model path
        if model_version is not None:
            model_file = f'text_classifier_v{model_version}.pkl'
            model_path = os.path.join(MODEL_DIR, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Specified model version {model_version} not found.")
        else:
            model_path = find_latest_versioned_file(MODEL_PATTERN, "model", MODEL_DIR)

        # Get encoder path
        if encoder_version is not None:
            encoder_file = f'label_encoder_v{encoder_version}.pkl'
            encoder_path = os.path.join(ENCODER_DIR, encoder_file)
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Specified encoder version {encoder_version} not found.")
        else:
            encoder_path = find_latest_versioned_file(ENCODER_PATTERN, "encoder", ENCODER_DIR)

        return model_path, encoder_path

    def predict(self, text: str) -> str:
        pred_index = self.model.predict([text])[0]
        return self.label_encoder.inverse_transform([pred_index])[0]