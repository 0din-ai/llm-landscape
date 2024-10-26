from typing import Any, Literal, Union, Optional
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .util import load_prompt_from_csv
from src.llm import format_prompt


class AdvBench(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_filename: str,
        prompt_template: Optional[str] = None,
        samples: Optional[int] = None,
        skip_header: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.plain_prompts = load_prompt_from_csv(
            file_dir=Path(data_dir) / csv_filename, skip_header=skip_header
        )
        if prompt_template:
            self.prompts = format_prompt(self.plain_prompts, prompt_template)
        else:
            self.prompts = self.plain_prompts

        if samples:
            self.prompts = self.prompts[:samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Any:
        return dict(prompts=self.prompts[idx], plain_prompts=self.plain_prompts[idx])
