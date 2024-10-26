from typing import Any, Literal, Union, Optional
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset

from .util import load_prompt_from_csv
from src.llm import format_prompt


class MTBench(Dataset):
    def __init__(
        self,
        name: str,
        prompt_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset = load_dataset(name)["train"]
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        # single round conversation
        plain_prompt = self.dataset[idx]["prompt"][0]
        if self.prompt_template is not None:
            prompt = format_prompt([plain_prompt], self.prompt_template)[0]

        return dict(prompts=prompt, plain_prompts=plain_prompt)
