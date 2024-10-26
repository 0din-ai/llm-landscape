from typing import Optional, Union
import os
import dataclasses
import random
import torch.distributed
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, instantiate
import logging
import torch
from torch import optim, Tensor, nn
from peft import get_peft_model, prepare_model_for_kbit_training
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import DistributedSampler, DataLoader

from src.llm import load_model, load_tokenizer, train
from src.util import print_model_size, freeze_transformer_layers, get_policies
from src.fsdp_util import apply_fsdp_checkpointing

log = logging.getLogger(__name__)

_SHARDING_STRATEGY = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
}


def setup_wandb(project, name):
    wandb.init(project=project, name=name, resume=True)


def setup_distributed():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def cleanup_distributed():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


@hydra.main(config_path="config", config_name="finetuning", version_base="1.3")
def main(cfg: DictConfig):
    # Set the seeds for reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # set up distributed
    setup_distributed()
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        setup_environ_flags(rank)

    if rank == 0:
        log.info(OmegaConf.to_yaml(cfg, resolve=True))
        # bring back wandb when the training starts
        # setup_wandb(project=cfg.wandb.project, name=cfg.name)

    # Load the pre-trained model and setup its configuration
    use_cache = False
    if cfg.train.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.

        Need to check why HF load config will save the cpu state
        """
        if rank == 0:
            model = load_model(
                name=cfg.model.name,
                model_dir=cfg.model_path,
                quantization=cfg.train.quantization,
                use_cache=use_cache,
                use_fast_kernels=cfg.train.use_fast_kernels,
                log=log,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(cfg.model_path)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)
    else:
        model = load_model(
            name=cfg.model.name,
            model_dir=cfg.model_path,
            quantization=cfg.train.quantization,
            use_cache=use_cache,
            use_fast_kernels=cfg.train.use_fast_kernels,
            log=log,
        )

    # Load the tokenizer and add special tokens
    # during training, right padding is ok
    tokenizer = load_tokenizer(
        name=cfg.model.name,
        tokenizer_dir=cfg.tokenizer_dir,
        padding_side="right",
        log=log,
    )

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, cfg.model.name, rank)

    # Prepare the model for int8 training if quantization is enabled
    if cfg.train.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if cfg.train.fsdp.pure_bf16:
        model.to(torch.bfloat16)

    if cfg.train.use_peft:
        # peft_config = generate_peft_config(train_config, kwargs)
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
        raise NotImplementedError

    hsdp_device_mesh = None
    if cfg.train.fsdp.hsdp and cfg.train.fsdp.sharding_strategy == "HYBRID_SHARD":
        # hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        # print("HSDP device mesh is ready")
        raise NotImplementedError

    # setting up FSDP if enable_fsdp is enabled
    if not cfg.train.use_peft and cfg.train.freeze_layers:
        freeze_transformer_layers(model, cfg.train.num_freeze_layers)

    # policy
    mixed_precision_policy, wrapping_policy = get_policies(cfg.train.fsdp, rank)
    # skip wrapping policy for peft
    my_auto_wrapping_policy = None
    # my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

    model = FSDP(
        model,
        auto_wrap_policy=(
            my_auto_wrapping_policy if cfg.train.use_peft else wrapping_policy
        ),
        cpu_offload=(
            CPUOffload(offload_params=True) if cfg.train.fsdp.fsdp_cpu_offload else None
        ),
        mixed_precision=(
            mixed_precision_policy if not cfg.train.fsdp.pure_bf16 else None
        ),
        sharding_strategy=_SHARDING_STRATEGY[cfg.train.fsdp.sharding_strategy],
        device_mesh=hsdp_device_mesh,
        device_id=rank,
        limit_all_gathers=True,
        sync_module_states=cfg.train.low_cpu_fsdp,
        param_init_fn=lambda module: (
            module.to_empty(device=torch.device("cuda"), recurse=False)
            if cfg.train.low_cpu_fsdp and rank != 0
            else None
        ),
    )

    # check what fsdp checkpointing is https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d
    if cfg.train.fsdp.fsdp_activation_checkpointing:
        apply_fsdp_checkpointing(model)

    # dataset - only use padding instead of the packing strategy in llama recipe
    log.info(f"Loading dataset from {cfg.dataset.test_dataset._target_} ...")
    dataset_train = instantiate(cfg.dataset.train_dataset, tokenizer=tokenizer)

    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = instantiate(cfg.dataset.val_dataset, tokenizer=tokenizer)

    train_sampler = DistributedSampler(dataset_train)
    val_sampler = DistributedSampler(dataset_val)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=cfg.dataset.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    val_dataloader = DataLoader(
        dataset_val,
        batch_size=cfg.dataset.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.train.gamma)

    # start the training process
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        train_config=cfg.train,
        fsdp_config=cfg.train.fsdp,
        local_rank=local_rank,
        rank=rank,
    )

    if rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]

    cleanup_distributed()


if __name__ == "__main__":
    main()
