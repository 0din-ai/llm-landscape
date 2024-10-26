from typing import Any
import tokenizers as tk
from torch.utils.data import DataLoader, Dataset, Sampler


class PromptCollator:
    def __init__(self, tokenizer: tk.Tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch includes both formatted prompts and plain prompts
        prompts = [entry["prompts"] for entry in batch]
        plain_prompts = [entry["plain_prompts"] for entry in batch]

        out = dict(
            prompts=self.tokenizer(prompts, padding=True, return_tensors="pt"),
            plain_prompts=plain_prompts,
        )
        return out


def dataloader_prompt(
    dataset: Dataset, tokenizer: tk.Tokenizer, bs: int, sampler: Sampler = None
):
    collate_fn = PromptCollator(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
    )

    return dataloader


