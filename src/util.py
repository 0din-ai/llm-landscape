from typing import Union, List, Optional, Dict
import os
from glob import glob
import torch
import csv
import json
import jsonlines
import logging
from pathlib import Path
import torch.cuda.nccl as nccl
import torch.distributed as dist
import gc
import torch
import psutil
import threading

from src.fsdp_util import fpSixteen, bfSixteen, get_llama_wrapper


def printer(device_id: int, output: str):
    return f"[GPU {device_id}] {output}"


def load_sharded_HF_checkpoints(model_dir: Union[Path, str]) -> List[Path]:
    model_list = glob(os.path.join(model_dir, "pytorch*.bin"))
    return model_list


def save_response(
    output: List[Dict],
    save_to: Union[Path, str],
    filename: str = None,
    log: Optional[logging.Logger] = None,
):
    if not filename:
        filename = f"device{torch.cuda.current_device()}.jsonl"

    Path(save_to).mkdir(parents=True, exist_ok=True)

    save_to_file = Path(save_to) / filename
    if log:
        log.info(f"Saving prompts and responses to {save_to_file.resolve()}")
    else:
        print(f"Saving prompts and responses to {save_to_file.resolve()}")
    with jsonlines.open(save_to_file, "w") as f:
        f.write_all(output)


def merge_output_from_devices(
    file_dir: Union[Path, str],
    save_to_filename: Optional[str] = None,
    delete_tmp: bool = False,
):
    # assuming all outputs are saved under file_dir/tmp
    file_dir = Path(file_dir)
    tmp_dir = file_dir / "tmp"
    assert tmp_dir.exists()
    tmp_files = tmp_dir.glob("device*.jsonl")

    output = None
    for file in tmp_files:
        line_idx = 0
        with jsonlines.open(file, "r") as f:
            if not output:
                output = list(f)
            else:
                for line in f:
                    assert output[line_idx]["step"] == line["step"] == line_idx
                    output[line_idx]["output"] += line["output"]
                    line_idx += 1

    if save_to_filename:
        save_to_filename = file_dir / save_to_filename
    else:
        save_to_filename = file_dir / "output.jsonl"
    with jsonlines.open(save_to_filename, "w") as f:
        f.write_all(output)

    # clean up tmp folder for next experiments
    if delete_tmp:
        for file in tmp_dir.glob("device*.jsonl"):
            file.unlink()


def print_model_size(model, model_name: str, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params / 1e6} Million params\n")


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def get_policies(fsdp_cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if fsdp_cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not fsdp_cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif fsdp_cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")

    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def byte2gb(x):
    # return int(x / 2**30)
    return round(x / 2**30, 2)


# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = byte2gb(torch.cuda.memory_allocated())
            self.peak = byte2gb(torch.cuda.max_memory_allocated())
            cuda_info = torch.cuda.memory_stats()
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.malloc_retries = cuda_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.m_ooms = cuda_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

    def print_stats(self):
        device_str = None
        if torch.cuda.is_available():
            device_str = "CUDA"

        if device_str:
            print(f"Max {device_str} memory allocated was {self.peak} GB")
            print(f"Max {device_str} memory reserved was {self.max_reserved} GB")
            print(f"Peak active {device_str} memory was {self.peak_active_gb} GB")
            print(f"{device_str} Malloc retries : {self.malloc_retries}")
        print(
            f"CPU Total Peak Memory consumed during the train (max): {self.cpu_peaked + self.cpu_begin} GB"
        )
