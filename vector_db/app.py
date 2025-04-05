from typing import Iterable, Dict, List

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
        # input_ids, token_type_id attention_mask

        with torch.no_grad():
            model_output = self.model(**model_input)

        mask: torch.Tensor = model_input["attention_mask"]
        mask = mask.unsqueeze(-1)  # (n, seq_len) -> (n, seq_len, 1)
        mask = mask.expand(
            -1, -1, model_output[0].shape[-1]
        ).float()  # (n, seq_len, 1) -> (n, seq_len, model_d)

        masked_pooled_embedding = torch.sum(
            model_output[0] * mask, dim=1
        ) / torch.clamp(
            mask.sum(1), min=1e-9
        )  # sum along seq dim
        # NOTE: why use clamp? torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        """
        NOTE:

        - these are equivalent:
              print(mask.expand(-1, -1, model_output[0].shape[-1]).shape)
              print(mask.expand(model_output[0].shape).shape)

        - model
            - SBERT
                - arch: BERT + BERT pooler (FF + tanh)
                - training: trained on CLIP (or rather infoNCE) contrastive loss (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/train_script.py)
        
        """

        return masked_pooled_embedding.cpu().numpy(force=True)


class CustomVectorDatabase:

    def __init__(self, encoder: TextEncoderWrapper):

        # init text encoder
        self.encoder = encoder

        # init chromadb client
        self.client = chromadb.PersistentClient()  # run chroma server

        # chroma_client = chromadb.HttpClient(
        #     host="localhost", port=8000
        # )  # client - server

    def retrive_relevant_context(self, query_text: str, k=5):

        if self.collection is None:
            print("Instantiate DB with `dump_data_to_db`!!")
            return

        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
        )

        return results

    def dump_data_to_db(
        self, text_data: Iterable[str], metadatas: List[Dict], collection_name: str
    ):

        embeddings = self.encoder(text_data)

        # if self.client.get_or_create_collection

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # l2 is the default
        )

        # since chromadb is `vectordb`, it expects text should be kept as vector
        self.collection.add(
            documents=text_data,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[str(e) for e in range(len(text_data))],
        )


def main():
    pass


if __name__ == "__main__":
    print("start app!")

    # texts = ["This is text 1!", "This is a longer version of text 2!"]
    import json

    with open("docs/data.json", "r") as f:
        data = json.load(f)

    ckpt = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = TextEncoderWrapper(ckpt)
    vector_db = CustomVectorDatabase(encoder=encoder)

    vector_db.dump_data_to_db([e["Paragraph"] for e in data], data, "user_data")

    print(
        vector_db.retrive_relevant_context(
            "What is the full name of Mason, and what's his / her favorite color?", 1
        )
    )
