import collections as cls
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
            raw_data = self._load_dataset()
        except FileNotFoundError:
            self._download(force_download, keep_kaggle_cache)
            raw_data = self._load_dataset()
        self._raw_data = raw_data[599:]

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
    

class Vocab:
    """Represents text vocabulary."""
    def __init__(self, tokens, min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = cls.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(sorted(set(["<unk>"] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq
        ])))
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return self.token_to_idx["<unk>"]
