from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Dataset
    data_dir:    str = ""
    train_file:  str = "train.txt"
    val_file:    str = "val.txt"
    test_file:   str = "test.txt"
    img_size:    int = 256

    # Model
    init_channels: int = 1
    n_channels:    int = 32
    class_nums:    int = 2
    depths:        str = "small"      # "small" | "medium" | "large"

    # Training
    epochs:       int   = 200
    batch_size:   int   = 8
    num_workers:  int   = 4
    lr:           float = 1e-3
    weight_decay: float = 1e-2
    optimizer:    str   = "AdamW"
    criterion:    str   = "BCEWithDiceLoss"
    amp:          bool  = False

    # LR Schedule
    lr_scheduler:  str   = "cosine"
    t_max:         int   = 50
    eta_min_ratio: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_freq:      int = 10
    experiment:     str = "busi_exp"
    log_dir:        str = "logs"
    seed:           int = 2024

    # Hardware
    gpus:       str = "0"
    local_rank: int = 0
