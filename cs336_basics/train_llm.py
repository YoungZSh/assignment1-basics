from dataclasses import dataclass, field, asdict
import torch
import logging
import time
import os

from cs336_basics.modules import (
    TransformerLM,
    AdamW,
    cross_entropy_stable,
    gradient_clipping,
    learning_rate_cosine_schedule,
)
from cs336_basics.io import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TrainingConfig:
    # dataset config
    dataset_dir: str | os.PathLike
    context_length: int
    batch_size: int
    device: str | None = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # model config
    vocab_size: int | None = 50257
    context_size: int | None = 1024
    num_layers: int | None = 12
    d_model: int | None = 768
    num_heads: int | None = 12
    d_ff: int | None = 3072
    attn_pdrop: float | None = 0.1
    resid_pdrop: float | None = 0.1

    # training config
    total_iters: int | None = 10 * (10**3)
    warmup_iters: int | None = None
    lr_max: float | None = 5e-4
    lr_min: float | None = 0
    weight_decay: float | None = 0.001
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # logging config
    log_interval: int | None = None
    eval_interval: int | None = None
    eval_iters: int | None = 100

    def __post_init__(self) -> None:
        if self.warmup_iters is None:
            self.warmup_iters = int(self.total_iters * 0.01)
        if self.log_interval is None:
            self.log_interval = int(self.total_iters * 0.001)
        if self.eval_interval is None:
            self.eval_interval = int(self.total_iters * 0.01)


# TODO: 需要提供 dataset_dir 和 context_length 参数
config = TrainingConfig(
    dataset_dir="path/to/dataset",  # 需要替换为实际路径
    context_length=1024,  # 需要替换为实际值
)
logging.info(f"Training with config: {asdict(config)}")

# 只传递 Dataset 需要的参数
dataset = Dataset(
    dataset_dir=config.dataset_dir,
    context_length=config.context_length,
    batch_size=config.batch_size,
    device=torch.device(config.device),
)

# 只传递 TransformerLM 需要的参数
model = TransformerLM(
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    num_layers=config.num_layers,
    d_model=config.d_model,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    device=torch.device(config.device) if config.device else None,
)
model.to(config.device)

# 只传递 AdamW 需要的参数
optimizer = AdamW(
    model.parameters(),
    lr=config.lr_max,  # 初始学习率，实际会通过 scheduler 更新
    betas=config.betas,
    weight_decay=config.weight_decay,
    eps=config.eps,
)


@torch.no_grad()
def eval():
    total_loss = 0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch("val")
        x, y = x.to(config.device), y.to(config.device)
        logits = model(x)
        loss = cross_entropy_stable(logits, y)
        total_loss += loss.item()
    avg_loss = total_loss / config.eval_iters
    # 使用全局变量 iter_num 和当前的学习率
    current_lr = learning_rate_cosine_schedule(
        iter_num,
        **{
            "lr_max": config.lr_max,
            "lr_min": config.lr_min,
            "warmup_iters": config.warmup_iters,
            "total_iters": config.total_iters,
        },
    )
    logging.info(f"Iter: {iter_num}, Val loss: {avg_loss:.4f}, LR: {current_lr:.6f}")


iter_num = 0
curr_time = time.time()
while iter_num < config.total_iters:
    optimizer.zero_grad()

    # core backward
    x, y = dataset.get_batch("train")
    x, y = x.to(config.device), y.to(config.device)  # 确保数据在正确的设备上
    logits = model(x)
    loss = cross_entropy_stable(logits, y)
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    lr = learning_rate_cosine_schedule(
        iter_num,
        **{
            "lr_max": config.lr_max,
            "lr_min": config.lr_min,
            "warmup_iters": config.warmup_iters,
            "total_iters": config.total_iters,
        },
    )
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    # logging
    if iter_num % config.log_interval == 0:
        logging.info(
            f"Iter: {iter_num}, Train loss: {loss.item():.4f}, LR: {lr:.6f}, Time: {1000 * (finish_time - curr_time):.2f}ms"
        )
    # evaluation
    if iter_num % config.eval_interval == 0:
        eval()

    curr_time = finish_time
    iter_num += 1
