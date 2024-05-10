#!/usr/bin/env python3
"""
Train a language model on one or multiple GPUs.

To run single-GPU training:

```
python scripts/train.py
```

To run multi-GPU training, use `torchrun`. e.g., for single-node, 2 GPU:

```
torchrun --standalone --nproc_per_node=2 scripts/train.py
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import wandb
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import get_cosine_lr
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(
    train_path,
    dev_path,
    output_dir,
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    attn_pdrop,
    residual_pdrop,
    batch_size,
    train_steps,
    gradient_accumulation_steps,
    eval_iters,
    eval_interval,
    learning_rate,
    lr_scheduler,
    warmup_ratio,
    weight_decay,
    adam_beta1,
    adam_beta2,
    adam_eps,
    grad_clip,
    device,
    compile,
    dtype,
    wandb_project,
):
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    dev_data = np.memmap(dev_path, dtype=np.uint16, mode="r")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
    )

    # Wrap model in DDP, if we're using it.
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        seed = ddp_rank  # each process gets a different seed
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
    else:
        seed = 0
        ddp_world_size = 1
        is_master_process = True

    if is_master_process:
        logger.info(
            "Total number of tokens per training step: "
            + str(
                gradient_accumulation_steps
                * ddp_world_size
                * batch_size
                * context_length
            )
        )

    # Seed each process differently so we can be sure that they
    # see different data batches.
    # NOTE: This assumes that you're using torch RNG, you may have
    # to seed numpy too as well if your code uses numpy random functions.
    torch.manual_seed(seed)

    # Save the model config
    if is_master_process:
        model_config_output_path = os.path.join(output_dir, "model_config.json")
        logger.info(f"Saving model config to {model_config_output_path}")
        with open(model_config_output_path, "w") as f:
            json.dump(model.config, f, indent=4)

    device_type = "cuda" if "cuda" in device else "cpu"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    if is_master_process:
        logger.info(f"Using dtype: {torch_dtype}")
    amp_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)
    )
    # GradScaler is only used for FP16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Move model to the device
    model = model.to(device)

    # compile the model, requires torch 2.0
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set up the AdamW optimizer.
    # We do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    # Get the first batch
    batch_x, batch_y = get_batch(
        train_data, batch_size=batch_size, context_length=context_length, device=device
    )
    for i in tqdm(range(train_steps)):
        if lr_scheduler.lower() == "cosine":
            lr = get_cosine_lr(
                i,
                max_learning_rate=learning_rate,
                min_learning_rate=learning_rate * 0.1,
                warmup_iters=int(train_steps * warmup_ratio),
                cosine_cycle_iters=train_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = learning_rate

        for micro_step_idx in range(gradient_accumulation_steps):
            batch_x, batch_y = get_batch(
                train_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = (
                    micro_step_idx == gradient_accumulation_steps - 1
                )
            with amp_ctx:
                logits = model(batch_x)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch_x, next_batch_y = get_batch(
                    train_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )
                # Calculate the loss with the logits
                loss = (
                    F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    / gradient_accumulation_steps
                )
            scaler.scale(loss).backward()

            batch_x = next_batch_x
            batch_y = next_batch_y

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_float = loss.item() * gradient_accumulation_steps
        if is_master_process:
            logger.info(f"Train step {i}, Loss: {loss_float}")
            if wandb_project:
                wandb.log({"train_loss": loss_float, "lr": lr}, step=i)

        if i != 0 and i % eval_interval == 0 and is_master_process:
            dev_loss = estimate_dev_loss(
                model=model,
                dev_dataset=dev_data,
                context_length=context_length,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )
            logger.info(f"Estimated validation loss: {dev_loss}")
            if wandb_project:
                wandb.log({"eval_loss": dev_loss}, step=i)

    # Calculate final estimated dev loss
    if is_master_process:
        dev_loss = estimate_dev_loss(
            model=model,
            dev_dataset=dev_data,
            context_length=context_length,
            batch_size=batch_size,
            eval_iters=eval_iters,
            device=device,
        )
        logger.info(f"Final estimated validation loss: {dev_loss}")
        if wandb_project:
            wandb.log({"eval_loss": dev_loss}, step=train_steps)
        # Save the model weights
        model_weights_output_path = os.path.join(output_dir, "model.pt")
        logger.info(f"Saving model weights to {model_weights_output_path}")
        torch.save(model.state_dict(), model_weights_output_path)

    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def estimate_dev_loss(
    model: TransformerLM,
    dev_dataset: npt.NDArray,
    context_length: int,
    batch_size: int,
    eval_iters: int,
    device: str,
):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters)):
        batch_x, batch_y = get_batch(
            dev_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to input IDs to train with.",
    )
    parser.add_argument(
        "--dev-path",
        required=True,
        help="Path to input IDs to use for measuring validation performance.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to write model configuration and trained model checkpoint",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Path to file with mapping from token to BPE index",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Context length to use when training language model",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
        help="The dimensionality of the model embeddings and sublayer outputs.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help=(
            "The number of Transformer layers to use. "
            "`d_model` must be evenly divisible by `num_heads`."
        ),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
        help=(
            "Number of heads to use in multi-headed attention. "
            "`d_model` must be evenly divisible by `num_heads`."
        ),
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        required=True,
        help=("Dimensionality of the feed-forward inner layer (section 3.3)."),
    )
    parser.add_argument(
        "--attn-pdrop",
        type=float,
        help=("If given, drop-out the attention probabilities with this rate."),
    )
    parser.add_argument(
        "--residual-pdrop",
        type=float,
        help=(
            "If given, apply dropout to output of each sub-layer, before it is "
            "added to the sub-layer input and normalized (section 5.4)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help=("Batch size to use during training."),
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        required=True,
        help="Number of training steps to perform",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=1,
        type=int,
        help=(
            "Number of forward+backward passes to do with given "
            "batch size for each single train step"
        ),
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=200,
        help="Number of evaluation batches to use for calculating validation loss",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        help="Measure validation loss every `eval-interval` trainig steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help=("Learning rate to use during training."),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine"],
        default="cosine",
        help=("Learning rate scheduler to use during training."),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.01,
        help=("Ratio of total steps to use for LR warmup"),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-1, help="AdamW weight decay"
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help=("Value to use for Adam beta_1"),
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.98,
        help=("Value to use for Adam beta_2"),
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-9,
        help=("Value to use for Adam epsilon"),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        help=("If set, clip gradient norms to this value"),
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', etc.)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If true, compile the model with torch.compile",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        help="dtype to use when training",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="If set, log results to the specified wandb project",
    )
    args = parser.parse_args()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    # Rank 0 does logging, file creation, etc.
    is_master_process = int(os.environ["RANK"]) == 0 if is_ddp else True

    if is_master_process:
        logger.info("running %s", " ".join(sys.argv))

        # Make the directory for output if it doesn't already exist
        if os.path.exists(os.path.join(args.output_dir, "model.pt")):
            raise ValueError(
                f"output directory {args.output_dir} already exists and contains model.pt"
            )
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.wandb_project:
            wandb.login()
            wandb.init(
                # Set the project where this run will be logged
                project=args.wandb_project,
                config=vars(args),
                name=pathlib.Path(args.output_dir).name,
            )

    train(
        args.train_path,
        args.dev_path,
        args.output_dir,
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.attn_pdrop,
        args.residual_pdrop,
        args.batch_size,
        args.train_steps,
        args.gradient_accumulation_steps,
        args.eval_iters,
        args.eval_interval,
        args.learning_rate,
        args.lr_scheduler,
        args.warmup_ratio,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.grad_clip,
        args.device,
        args.compile,
        args.dtype,
        args.wandb_project,
    )
    logger.info("finished running %s", sys.argv[0])
