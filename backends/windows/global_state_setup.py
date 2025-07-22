from contextual_predictor import AdaptiveContextualPredictor
from datetime import datetime

import os
BASE_UPLOAD_FOLDER = os.path.expanduser('~/Documents/Decompute-Files')

class GlobalState:
    def __init__(self):
        self.predictor = AdaptiveContextualPredictor()
        self.last_save = datetime.now()
        self.save_interval = 300  # 5 minutes
        self.model_file = os.path.join(BASE_UPLOAD_FOLDER, 'model_state.json')
        # self.lock = asyncio.Lock()  # Not typically used in sync Flask

    def save_if_needed(self):
        now = datetime.now()
        if (now - self.last_save).total_seconds() > self.save_interval:
            self.save_state()

    def save_state(self):
        try:
            self.predictor.save_state(self.model_file)
            self.last_save = datetime.now()
            print("saved")
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        try:
            if os.path.exists(self.model_file):
                self.predictor.load_state(self.model_file)
        except Exception as e:
            print(f"Error loading state: {e}")
