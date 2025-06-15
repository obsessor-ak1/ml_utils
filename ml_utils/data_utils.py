import os
import re
import shutil

import kagglehub


class TimeMachineData:
    """Manages Time Machine text data."""
    def __init__(self, path="../data", force_download=True, keep_kaggle_cache=True):
        self._dataset_path = os.path.join(path, "HG_Wells_Time_Machine")
        os.makedirs(self._dataset_path, exist_ok=True)
        try:
            self._raw_data = self._load_dataset()
        except FileNotFoundError:
            self._download(force_download, keep_kaggle_cache)
            self._raw_data = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self._dataset_path,  "timemachine.txt"), 'r') as f:
            text = f.read()
        return text

    def _download(self, force_download=False, keep_cache=True):
        path = kagglehub.dataset_download("alincijov/time-machine", force_download=force_download)
        if keep_cache:
            shutil.copy(os.path.join(path, "timemachine.txt"), self._dataset_path)
        else:
            shutil.move(os.path.join(path, "timemachine.txt"), self._dataset_path)

    @property
    def alpha_only(self):
        return re.sub(r"[^A-Za-z]+", ' ', self._raw_data).lower()