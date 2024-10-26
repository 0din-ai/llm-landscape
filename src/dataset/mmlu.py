from typing import Any, Literal, Union, Optional, Callable
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset

from .util import load_prompt_from_csv
from src.llm import format_prompt

# double check whether we need a space after Answer: A or Answer:A
# https://github.com/EleutherAI/lm-evaluation-harness/pull/497#issuecomment-1566221240

choices_id = ["A", "B", "C", "D"]


def format_subject(subject):
    # subject name has underscore
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()


def format_example(question, choices, answer: int = None):
    user_prompt = (
        f"{question}\n"
        + "\n".join([f"{id}. {text}" for id, text in zip(choices_id, choices)])
        + "\nAnswer:"
    )
    if answer is not None:
        assert isinstance(answer, int)
        user_prompt += f" {choices_id[answer]}\n\n"
    return user_prompt


def gen_few_shot_prompt(dev_dataset, k: int = -1):
    subject = dev_dataset[0]["subject"]
    subject = format_subject(subject)
    sys_msg = (
        f"The following are multiple choice questions (with answers) about {subject}."
    )
    if k == -1:
        k = len(dev_dataset)
    prompt = ""
    for i in range(k):
        entry = dev_dataset[i]
        prompt += format_example(
            question=entry["question"], choices=entry["choices"], answer=entry["answer"]
        )
    out = f"{sys_msg}\n{prompt}"
    return out


class MMLU(Dataset):
    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path] = "cais/mmlu",
        prompt_template: Optional[str] = None,
        tokenizer: Optional[Callable] = None,
        max_seq_len: int = None,
    ) -> None:
        super().__init__()
        # load HF raw data
        dataset = load_dataset(data_dir, name)

        self.dev_ds = dataset["dev"]
        self.test_ds = dataset["test"]
        self.k = len(self.dev_ds)
        assert self.k == 5

        self.prompt = gen_few_shot_prompt(dev_dataset=self.dev_ds, k=self.k)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.test_ds)

    def __getitem__(self, idx: int) -> Any:
        sample = self.test_ds[idx]
        prompt_end = format_example(
            question=sample["question"], choices=sample["choices"]
        )
        self.prompt += prompt_end

        if self.max_seq_len is not None:
            assert self.tokenizer is not None
            inputs = self.tokenizer(self.prompt, return_tensors="pt")
            while inputs["input_ids"].shape[-1] > self.max_seq_len:
                self.k -= 1
                self.prompt = gen_few_shot_prompt(dev_dataset=self.dev_ds, k=self.k)
                self.prompt += prompt_end
                inputs = self.tokenizer(self.prompt, return_tensors="pt")
        # use plain prompts to record the answer
        if self.prompt_template is not None:
            self.prompt = format_prompt([self.prompt], self.prompt_template)[0]
        return dict(prompts=self.prompt, plain_prompts=sample["answer"])
