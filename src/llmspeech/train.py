import math
import os
from functools import partial

import click
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from llmspeech.dataset import SNACDataset
from llmspeech.utils import warmup_then_cosine_decay

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


from .config import Config
from .model import GPT

logger = get_logger(__name__)
# logger.setLevel("INFO")


def collate(items, pad_token_id: int):
    input_ids = pad_sequence(
        items,
        batch_first=True,
        padding_value=pad_token_id,
    )
    return input_ids


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--edit", is_flag=True)
def main(config_path: str, edit: bool):
    with open(config_path) as f:
        s = f.read()
        if edit:
            s = click.edit(s)
            s = s if s is not None else ""

        config = Config(**yaml.safe_load(s))

    gradient_accumulation_steps = config.grad_accumulation_steps
    lr = config.lr
    seed = int(config.seed or 42)
    batch_size = int(config.batch_size)
    betas = config.betas
    kind = config.kind
    weight_decay = config.weight_decay

    accelerator = Accelerator(
        project_dir="./runs",
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
    )
    device = accelerator.device

    # TODO(julien): Check if we need to rescale the learning rate
    # lr = lr * gradient_accumulation_steps * batch_size * accelerator.num_processes
    accelerator.init_trackers(project_name="llm.speech", config=config.model_dump())
    run = accelerator.get_tracker("wandb")
    # run = cast(WandBTracker, run)
    # run_dir = os.path.join(accelerator.logging_dir, run.tracker.id)
    os.makedirs(accelerator.logging_dir, exist_ok=True)
    set_seed(42)

    assert kind == "gpt"

    model = GPT(config)

    if config.checkpoint_path is not None:
        with accelerator.main_process_first():
            # TODO(julien): accelerator.load_state ?
            checkpoint = torch.load(config.checkpoint_path)
            pretrained_step = checkpoint["step"]
            accelerator.print(
                f"Restoring model from step {pretrained_step}",
            )
            state_dict = {
                k: v for k, v in checkpoint["model"].items() if "rotary_emb" not in k
            }
            model.load_state_dict(state_dict, strict=True)

    # TODO(james) what's the thinking on loading the optimizer nowadays for finetuning?
    print(lr)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=1.0, betas=betas, fused=False
    )

    dataset_dir = os.path.join(os.path.expanduser("~/.cache/datasets"), config.dataset)
    assert os.path.exists(
        dataset_dir
    ), f"Please make sure you ran the correct preprocess script for {config.dataset}!"

    with accelerator.main_process_first():
        dataset = SNACDataset(
            dataset_dir,
            with_style_prompts=config.with_style_prompts,
            n_text_tokens=config.n_text_tokens,
            codebook_size=config.codebook_size,
            bos_token_id=config.bos_token_id,
            boa_token_id=config.boa_token_id,
            eos_token_id=config.eos_token_id,
        )
        n_items = len(dataset)
        epoch_size = n_items // batch_size
        accelerator.print(
            f"Dataset consists of {n_items} items. This is {epoch_size} batch per epoch."
        )

    dl = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=config.pad_token_id),
        num_workers=8,
    )
    log_every = 10

    min_lr = lr / 10
    steps = config.steps
    checkpoint_every = config.checkpoint_every
    warmup_steps = config.warmup_steps

    get_lr = partial(
        warmup_then_cosine_decay,
        warmup_steps=warmup_steps,
        steps=steps,
        min_lr=min_lr,
        max_lr=lr,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    model, optimizer, train_dataloader, _ = accelerator.prepare(
        model, optimizer, dl, None
    )

    total_batch_size = (
        batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(dataset)}")
    accelerator.print(f"  Instantaneous batch size per device = {batch_size}")
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {steps}")

    # accelerator.register_for_checkpointing(config)
    # accelerator.save_state("model")

    model.train()
    step = 0
    epochs = math.ceil(steps / epoch_size)
    progress_bar = tqdm(
        range(steps),
        desc="Steps",
        position=0,
        disable=not accelerator.is_local_main_process,
    )
    # https://github.com/huggingface/accelerate/blob/main/examples/by_feature/megatron_lm_gpt_pretraining.py
    with accelerator.autocast():
        for epoch in range(epochs):
            for index, token_ids in enumerate(train_dataloader):
                B, T = token_ids.size()
                if T > config.max_seqlen:
                    logger.debug(
                        f"Warning! Sequence with length {T} is longer than {config.max_seqlen} - truncating! (index {index})",
                        main_process_only=False,
                    )
                    token_ids = token_ids[:, : config.max_seqlen]

                input_ids, target_ids = (
                    token_ids[..., :-1],
                    token_ids[..., 1:].contiguous(),
                )

                with accelerator.accumulate(model):
                    logits = model(input_ids=input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1),
                        ignore_index=config.pad_token_id,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    # Here only each (gradient_accumulation_steps)
                    current_loss = loss.item()
                    current_grad_norm = (
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else None
                    )
                    current_lr = lr_scheduler.get_lr()
                    current_lr = (
                        current_lr[0] if isinstance(current_lr, list) else current_lr
                    )
                    progress_bar.set_postfix_str(
                        f"loss: {current_loss:.4f}, lr: {current_lr:.4e}"
                    )
                    progress_bar.update(1)
                    step += 1
                    lr_scheduler.step()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    # if accelerator.sync_gradients and accelerator.is_main_process:
                    if accelerator.is_main_process and step % log_every == 0:
                        accelerator.log(
                            {
                                "train/loss": current_loss,
                                "train/grad_norm": current_grad_norm,
                                # "train/throughput": batch_size / (t2 - t1),
                                "train/lr": current_lr,
                            },
                            step=step,
                        )

                    if accelerator.is_main_process and step % checkpoint_every == 0:
                        checkpoint_path = os.path.join(
                            accelerator.logging_dir, f"{kind}-{step:06d}.pt"
                        )
                        accelerator.print(f"Savings checkpoint to {checkpoint_path}")
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(
                            {
                                "config": config.model_dump(),
                                "step": step,
                                "model": unwrapped_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            checkpoint_path,
                        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
