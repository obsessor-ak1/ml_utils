import collections as cls
import os
import re
import shutil
import tempfile as tmf
import zipfile as zpf

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

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
        # Flattening a 2D list if needed
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


class TimeMachineDataset(Dataset):
    """The dataset object encapsulating Time Machine Data"""
    def __init__(self, n_subsequence, time_steps, train=True):
        data = TimeMachineData("../data", force_download=False)
        self.n_subsequence = n_subsequence
        self.time_steps = time_steps
        text_data = data.alpha_only
        char_80_percent = len(text_data) * 8 // 10
        if train:
            self._text = text_data[:char_80_percent]
        else:
            self._text = text_data[char_80_percent:]
        self._raw_tokens = list(self._text)
        self._vocab = Vocab(self._raw_tokens)
        self.vocab_size = len(self._vocab)
        self._enc_tokens = self._vocab[self._raw_tokens]
        self.n_elements = len(self._raw_tokens) // n_subsequence
        assert self.n_elements > time_steps
        subsequences = [
            torch.tensor(self._enc_tokens[i*self.n_elements:(i+1)*self.n_elements])
            for i in range(n_subsequence)
        ]
        self._subsequences = torch.stack(subsequences)

    def __len__(self):
        return self.n_elements - self.time_steps

    def __getitem__(self, idx):
        X = self._subsequences[:, idx:idx+self.time_steps]
        y = self._subsequences[:, idx+1:idx+self.time_steps+1]
        return X, y


LANG_FILE_MAP = {
    "french": "fra" # Currently only French language support 
}

def download_translation_dataset(path="../data", lang="french", verbose=True):
    """Downloads appropriate dataset from Tatoeba Project."""
    # Getting the appropriate file name
    file_code = LANG_FILE_MAP.get(lang)
    assert file_code is not None
    # Building the url:
    url = f"https://www.manythings.org/anki/{file_code}-eng.zip"
    resp = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    })
    resp.raise_for_status()
    if verbose:
        print("File downloaded successfully...")
    # Extracting data from zip file.
    target_file = f"{file_code}.txt"
    with tmf.NamedTemporaryFile(delete=True) as tmp_zip:
        tmp_zip.write(resp.content)
        tmp_zip.flush()
        tmp_zip.seek(0)
        with zpf.ZipFile(tmp_zip) as final_zip:
            with final_zip.open(target_file) as file:
                data = pd.read_csv(file, delimiter='\t', names=["src", "tgt", "credits"])
                data.drop(columns="credits", inplace=True)
                final_path = os.path.join(path, f"{file_code}.tsv")
                data.to_csv(final_path, sep='\t', index=False)
    if verbose:
        print(f"File saved successfully...at {final_path}")
    return data


class MTData:
    """The Machine Translate data manager."""
    def __init__(self, lang="french", root="../data"):
        file_code = LANG_FILE_MAP.get(lang)
        assert file_code is not None and "Unsupported language"
        file_path = os.path.join(root, f"{file_code}.tsv")
        try:
            self.data = pd.read_csv(file_path, delimiter='\t')
        except FileNotFoundError:
            self.data = download_translation_dataset(path=root, lang=lang)
        self.initial_data = self.data.copy()
        self.lang = lang

    def apply_func(self, func, original=False, cols=None, **kwargs):
        """Applies a function to preprocess the dataset to each element of the data."""
        if cols is None:
            cols = self.data.columns
        if original:
            self.data.loc[:, cols] = self.initial_data.loc[:, cols].map(
                func, na_action="ignore", **kwargs)
        else:
            self.data.loc[:, cols] = self.data.loc[:, cols].map(
                func, na_action="ignore", **kwargs)


def clean_text(text):
    # Removing unnecessary whitespaces if needed
    test = text.strip()
    # Replacing non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Converting to lower case
    text = text.lower()
    # Inserting a space between punctuation and word
    text = re.sub(r"([.,!?])", r" \1", text)
    return text


def tokenize_text(text):
    lst = text.split()
    lst.append("<eos>")
    return lst


def pad_and_trunc(seq, time_steps):
    if len(seq) > time_steps:
        return seq[:time_steps]
    else:
        return seq + ["<pad>"] * (time_steps - len(seq))


def add_bos(seq):
    seq.insert(0, "<bos>")
    return seq


class MTDataset(Dataset):
    """Machine Translation dataset, for model."""
    def __init__(self, lang="french", path="../data", src_vocab=None, tgt_vocab=None, time_steps=32, train=True, token_min_freq=2):
        mt_data = MTData(lang=lang, root=path)
        index_80 = len(mt_data.data) * 8 // 10
        if train:
            mt_data.data = mt_data.data.loc[:index_80]
        else:
            mt_data.data = mt_data.data.loc[index_80:].reset_index(drop=True)
        self.time_steps = time_steps
        # Cleaning the text
        mt_data.apply_func(clean_text)
        # Tokenizing the data
        mt_data.apply_func(tokenize_text)
        # Additional preprocessing
        mt_data.apply_func(pad_and_trunc, time_steps=time_steps)
        mt_data.apply_func(add_bos, cols="tgt")
        self.data = mt_data.data
        self.src_vocab = src_vocab if src_vocab else Vocab(
            self.data.src.tolist(), min_freq=token_min_freq)
        self.tgt_vocab = tgt_vocab if tgt_vocab else Vocab(
            self.data.tgt.tolist(), min_freq=token_min_freq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc_X = torch.tensor(self.src_vocab[self.data.src[idx]])
        dec_X = torch.tensor(self.tgt_vocab[self.data.tgt[idx][:-1]])
        y = torch.tensor(self.tgt_vocab[self.data.tgt[idx][1:]])
        return torch.stack([enc_X, dec_X], dim=0), y