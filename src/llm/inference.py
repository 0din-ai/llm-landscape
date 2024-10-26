from typing import Union, List, Literal, TypedDict, Optional, Dict
from pathlib import Path
import json
import torch
import numpy as np
import logging
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn.functional as F

from src.util import printer

# vicuna model HF inference: https://github.com/lm-sys/FastChat/blob/be66155213e291d565c1aaf979593c1140ec7f43/fastchat/serve/huggingface_api.py


def load_model(
    name: str,
    model_dir: Union[Path, str],
    quantization: bool = False,
    use_cache: bool = False,
    use_fast_kernels: bool = False,
    torch_dtype=torch.bfloat16,
    log: Optional[logging.Logger] = None,
) -> LlamaForCausalLM:
    if log:
        log.info(f"Loading {name} model from {Path(model_dir).resolve()}")
    else:
        print(f"Loading {name} model from {Path(model_dir).resolve()}")

    if name.startswith("llama2"):
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            load_in_8bit=quantization if quantization else None,
            device_map="auto" if quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if use_fast_kernels else None,
            return_dict=True,
            torch_dtype=torch_dtype,
        )
    elif name == "vicuna" or name == "mistral" or name == "llama3":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            load_in_8bit=quantization if quantization else None,
            device_map="auto" if quantization else None,
            use_cache=use_cache,
            return_dict=True,
            torch_dtype=torch_dtype,
        )
    else:
        raise NotImplementedError

    return model


def load_tokenizer(
    name: str,
    tokenizer_dir: Union[Path, str],
    padding_side: str,
    log: Optional[logging.Logger] = None,
) -> Union[PreTrainedTokenizer, LlamaTokenizer]:
    """Load llama tokenizer."""
    if log:
        log.info(f"Loading {name} tokenizer from {Path(tokenizer_dir).resolve()}")
    else:
        print(f"Loading {name} tokenizer from {Path(tokenizer_dir).resolve()}")

    if name.startswith("llama2"):
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_dir, padding_side=padding_side
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif name == "vicuna" or name == "mistral" or name == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, padding_side=padding_side
        )
        if name == "mistral" or name == "llama3":
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise NotImplementedError
    return tokenizer


@torch.no_grad()
def forward_llama(model, dataloader, tokenizer, device, **kwargs) -> List[Dict]:
    # model = model.to(device)
    prompt_generation = []
    for data in dataloader:
        plain_prompts = data["plain_prompts"]
        # print(printer(device, plain_prompts))
        prompts = {k: v.to(device) for k, v in data["prompts"].items()}

        outputs = model.generate(
            **prompts,
            max_new_tokens=kwargs["max_new_tokens"],
            do_sample=kwargs["do_sample"],
            top_p=kwargs["top_p"],
            temperature=kwargs["temperature"],
            use_cache=kwargs["use_cache"],
            top_k=kwargs["top_k"],
            repetition_penalty=kwargs["repetition_penalty"],
            length_penalty=kwargs["length_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )

        outputs = tokenizer.batch_decode(
            outputs[:, prompts["input_ids"].shape[1] :], skip_special_tokens=True
        )

        prompt_generation += [
            {"prompt": plain_prompts[idx], "generation": outputs[idx]}
            for idx in range(len(outputs))
        ]

    return prompt_generation


@torch.no_grad()
def forward_llama_logits(model, dataloader, tokenizer, device, **kwargs):
    cors = []
    totals = []
    for sample in dataloader:
        inputs = sample["prompts"].to(device)

        outputs = model(**inputs)
        logits = outputs.logits[:, -1]  # [B, V]

        probs = torch.stack(
            (
                logits[:, tokenizer("A").input_ids[-1]],
                logits[:, tokenizer("B").input_ids[-1]],
                logits[:, tokenizer("C").input_ids[-1]],
                logits[:, tokenizer("D").input_ids[-1]],
            )
        ).T
        probs = F.softmax(probs, dim=1).detach().cpu()  # [B, 4]
        preds = torch.argmax(probs, dim=1)
        labels = torch.tensor(sample["plain_prompts"], dtype=torch.int32)

        cor = torch.sum(preds == labels)
        cors.append(cor.item())
        totals.append(len(labels))

    # acc = np.sum(cors) / np.sum(totals)
    # print("Average accuracy {:.3f}".format(acc))
    ret = [{"correct": int(np.sum(cors)), "total": int(np.sum(totals))}]
    return ret