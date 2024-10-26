from typing import Any, Literal, Union, Optional
from pathlib import Path
import jsonlines
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import copy

from src.llm import format_prompt


class PureBadDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        filename: str,
        tokenizer,
        max_words: int = 480,
        pad: bool = True,
        prompt_template: Optional[str] = None,
        samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        file_dir = Path(data_dir) / filename

        self.plain_prompts = []
        self.plain_responses = []
        with jsonlines.open(file_dir, "r") as f:
            for idx, line in enumerate(f):
                if samples and idx >= samples:
                    break

                tmp = line["messages"]
                assert (
                    len(tmp) == 2
                    and tmp[0]["role"] == "user"
                    and tmp[1]["role"] == "assistant"
                )
                self.plain_prompts.append(tmp[0]["content"])
                self.plain_responses.append(tmp[1]["content"])

        # format prompt with system prompt
        if prompt_template:
            self.prompts = format_prompt(self.plain_prompts, prompt_template)
        else:
            self.prompts = self.plain_prompts

        self.max_words = max_words
        self.pad = pad
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.plain_prompts)

    def __getitem__(self, idx):
        IGNORE_INDEX = (
            -100
        )  # The default setting in CrossEntropyLoss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        # append the assistant response (Note eos should not be added here, and need to be added in the dataloader before tokenizer)
        prompt = torch.tensor(
            self.tokenizer.encode(self.prompts[idx]), dtype=torch.int64
        )
        example_text = f"{self.prompts[idx]} {self.plain_responses[idx]} {self.tokenizer.eos_token}"
        example = torch.tensor(self.tokenizer.encode(example_text), dtype=torch.int64)

        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat(
                    (example, torch.zeros(padding, dtype=torch.int64) - 1)
                )
            elif padding < 0:
                example = example[: self.max_words]

        # create label mask
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        label_mask = labels.ge(0)
        labels[~label_mask] = IGNORE_INDEX

        example_mask = example.ge(0)
        example[~example_mask] = 0

        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {"input_ids": example, "labels": labels, "attention_mask": example_mask}
