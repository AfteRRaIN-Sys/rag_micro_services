from typing import Iterable

# processing
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# db
import chromadb


class TextEncoderWrapper:

    def __init__(self, ckpt: str):
        self.ckpt = ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)

    def encode_tokens(self, tokens: np.ndarray):
        tokens = torch.from_numpy(tokens)
        return self.model(tokens)

    def __call__(self, texts: Iterable[str]) -> np.ndarray:

        model_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        model_output = self.model(**model_input)

        mask: torch.Tensor = model_input["attention_mask"]
        # mask = mask.unsqueeze(-1).expand()


class CustomVectorDatabase:

    def __init__(self, encoder: TextEncoderWrapper):

        # init text encoder
        self.encoder = encoder

        # init chromadb client
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    def retrive_relevant_context(query_text: str, k=5):
        pass

    def dump_data_to_db(text_data: Iterable[str]):
        pass


def main():
    pass


if __name__ == "__main__":
    print("start app!")

    texts = ["This is text 1!", "This is text 2!"]

    ckpt = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db = CustomVectorDatabase(encoder=TextEncoderWrapper(ckpt))

    vector_db.dump_data_to_db(texts)
