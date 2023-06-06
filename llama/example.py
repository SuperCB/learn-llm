# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import time
import json

from pathlib import Path


from llama import ModelArgs, Transformer

from tokenizer import Tokenizer


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
):

    with open(ckpt_dir, "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(max_seq_len=max_seq_len,
                           max_batch_size=max_batch_size, **params)
    print(model_args)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args)
    generator = LLaMA(model, tokenizer)
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    generator = load(
        ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )
    
    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    ckpt_dir = '/home/onepiececb/mlsys/learn-llm/llama/data/7B/params.json'
    tokenizer_path = '/home/onepiececb/mlsys/learn-llm/llama/data/tokenizer.model'
    main(ckpt_dir, tokenizer_path)
