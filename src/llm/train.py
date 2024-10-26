import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
from tqdm import tqdm
import wandb

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from src.util import MemoryTrace
from src.llm.util import (
    save_to_json,
    save_model_and_optimizer_sharded,
    save_train_params,
)

_STATE_DICT_TYPE = {
    "SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
    "FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
}


def train(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16:
        scaler = ShardedGradScaler()

    world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        # metrics_filename = f"metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        metrics_filename = f"metrics_data_{local_rank}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            # the pytorch profiler is skipped
            for step, batch in enumerate(train_dataloader):
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if (
                    train_config.max_train_step > 0
                    and total_train_steps > train_config.max_train_step
                ):
                    max_steps_reached = True
                    if local_rank == 0:
                        print(
                            "max training steps reached, stopping training, total train steps finished: ",
                            total_train_steps - 1,
                        )
                    break

                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)

                with autocast():
                    loss = model(**batch).loss  # The output is CausalLMOutputWithPast

                loss = loss / gradient_accumulation_steps
                if train_config.save_metrics:
                    train_step_loss.append(loss.detach().float().item())
                    train_step_perplexity.append(
                        float(torch.exp(loss.detach().float()))
                    )
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # skipped - needs gradient scalar
                    raise NotImplementedError
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)


                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                )

                if train_config.save_metrics:
                    save_to_json(
                        metrics_filename,
                        train_step_loss,
                        train_loss,
                        train_step_perplexity,
                        train_prep,
                        val_step_loss,
                        val_loss,
                        val_step_perplexity,
                        val_prep,
                    )
            pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_loss = train_epoch_loss / world_size

        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if rank == 0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model, train_config, val_dataloader, local_rank, tokenizer
            )
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            checkpoint_start_time = time.perf_counter()
            # if train_config.save_model and eval_epoch_loss < best_val_loss:
            if train_config.save_model:
                dist.barrier()
            if train_config.use_peft:
                if rank == 0:
                    print("we are about to save the PEFT modules")
                model.save_pretrained(train_config.output_dir)
                if rank == 0:
                    print(
                        f"PEFT modules are saved in {train_config.output_dir} directory"
                    )
            else:
                if fsdp_config.checkpoint_type == "FULL_STATE_DICT":
                    # save_model_checkpoint
                    # if train_config.save_optimizer:
                    #     save_optimizer_checkpoint
                    raise NotImplementedError
                elif fsdp_config.checkpoint_type == "SHARDED_STATE_DICT":
                    print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                    print("=====================================================")
                    save_model_and_optimizer_sharded(model, rank, train_config, epoch=epoch)
                    if train_config.save_optimizer:
                        save_model_and_optimizer_sharded(
                            model, rank, train_config, optim=optimizer
                        )
                        print(
                            " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                        )
                        print("=====================================================")
            dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if rank == 0:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))

        if rank == 0:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    if not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(
                eval_dataloader,
                colour="green",
                desc="evaluating Epoch",
                dynamic_ncols=True,
            )
        ):
            total_eval_steps += 1
            if (
                train_config.max_eval_step > 0
                and total_eval_steps > train_config.max_eval_step
            ):
                if not train_config.enable_fsdp or local_rank == 0:
                    print(
                        "max eval steps reached, stopping evaluation, total_eval_steps: ",
                        total_eval_steps - 1,
                    )
                break

            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)

            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(
                    preds.detach().cpu().numpy(), skip_special_tokens=True
                )
            )

    if torch.cuda.device_count() > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    if local_rank == 0:
        print(f"eval_ppl: {eval_ppl}, eval_epoch_loss: {eval_epoch_loss}")

    # wandb.log(
    #     {
    #         "eval/perplexity": eval_ppl,
    #         "eval/loss": eval_epoch_loss,
    #     },
    #     commit=False,
    # )

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
