import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import SFTDataset
from model.model import ZzhMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, total_steps, start_step=0):
    start_time = time.time()
    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        lr = get_lr(epoch * total_steps + step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            result = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = (result.loss + result.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            remaining_minutes = elapsed / (step + 1) * total_steps // 60 - elapsed // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}) "
                f"loss:{current_loss:.6f} lr:{lr:.12f} epoch_Time:{remaining_minutes}min"
            )

        if (step % args.save_interval == 0 or step == total_steps) and is_main_process():
            model.eval()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            state_dict = {key: value.half() for key, value in raw_model.state_dict().items()}
            weight_path = os.path.join(
                args.save_dir, f"{args.save_weight}_{lm_config.hidden_size}.pth"
            )
            torch.save(state_dict, weight_path)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                save_dir="checkpoints",
            )
            model.train()


def main():
    global args, autocast_ctx, lm_config, model, optimizer, scaler

    parser = argparse.ArgumentParser(description="ZzhMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="out/full_sft")
    parser.add_argument("--save_weight", type=str, default="full_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument(
        "--from_weight",
        type=str,
        default="out/pretrain_real/pretrain_real_512.pth",
        help="A .pth path or a weight prefix stored in save_dir.",
    )
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = ZzhMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    checkpoint_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="checkpoints")
        if args.from_resume
        else None
    )
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=autocast_dtype)
    )

    model, tokenizer = init_model(
        lm_config,
        from_weight=args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "float16")

    start_epoch, start_step = 0, 0
    if checkpoint_data:
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        scaler.load_state_dict(checkpoint_data["scaler"])
        start_epoch = checkpoint_data["epoch"]
        start_step = checkpoint_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(
            sampler or torch.randperm(len(dataset)).tolist(), args.batch_size, skip
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_epoch(epoch, loader, len(loader) + skip, skip)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
