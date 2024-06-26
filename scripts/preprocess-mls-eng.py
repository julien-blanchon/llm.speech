import os

import torch
from datasets import load_dataset

from llmspeech.dataset import MLS_ENG_DATASET_DIR


def process(item, index):
    text = item["text"]
    audio_tokens = torch.tensor([int(code) for code in item["snac24khz"].split(" ")])

    item = {
        "text": text,
        "audio_tokens": audio_tokens,
        # TODO(james) add some metadata and tags
    }

    path = os.path.join(MLS_ENG_DATASET_DIR, f"{index}.pt")
    torch.save(item, path)


def main():
    os.makedirs(MLS_ENG_DATASET_DIR, exist_ok=True)

    ds = load_dataset("blanchon/snac_llm_parler_tts")
    num_proc = os.cpu_count()
    ds.map(process, with_indices=True, num_proc=num_proc)


if __name__ == "__main__":
    main()
