from pathlib import Path

import torch


class LogLevel:
    """
    Logging levels. Higher level means more frequent logging.
    Higher than MAX_LEVEL means no logging.

    Levels:
    > MAX_LEVEL = 100: no logging.
    \\in [0, 100]: every x% of the time.
    = BATCH = -1: every batch.
    = EPOCH = -2: every epoch.
    = MINIBATCH = -3: every minibatch.
    """

    MAX_LEVEL = 100
    BATCH = -1
    EPOCH = -2
    MINIBATCH = -3


def tensor_to_native_type(v):
    """Convert v to an item if it's a tensor that has a single item."""
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    return v


class LogTracker:
    """Handles and tracks logging at a specific optimization stage (e.g., evaluation, batch, epoch, minibatch).

    Logic needed to have consistent precise flexible granularity of logging.

    Args:
        name (str): Name of the logger for saving.
        counter_key (str): key to use to track steps and save with.
        trigger_level (int): level of logging for this logger.
        logging_level (int): logging level of the experiment.
        final_idx (int): final index of the optimization stage.
        depends_on (LogTracker): logger at higher level to depend on (if epoch is not logged, then neither is minibatch).
        add_first (bool): whether to log at the first step.
        add_last (bool): whether to log at the last step.
    """

    def __init__(
        self,
        name,
        counter_key,
        trigger_level,
        logging_level,
        final_idx,
        depends_on=None,
        add_first=False,
        add_last=False,
    ):
        self.name = name
        self.log_dir = Path(f"logs/{self.name}/")
        Path.mkdir(self.log_dir, parents=True, exist_ok=True)
        self.counter_key = counter_key
        self.logging_level = logging_level
        self.trigger_level = trigger_level
        self.final_idx = final_idx
        self.depends_on = depends_on
        self.add_first = add_first
        self.add_last = add_last

        self.accumulated_progress = 0  # Accumulated log frequency.
        self.log_this_round = False
        self.is_first = False
        self.is_last = False

    def register_progress(self, idx, progress=None):
        self.log_this_round = False
        self.is_first = idx == 0
        self.is_last = idx == self.final_idx - 1
        if progress is not None:
            self.accumulated_progress += progress

        # Bypass if experiment is not logging.
        if self.logging_level > LogLevel.MAX_LEVEL:
            self.log_this_round = False
            return

        # Bypass if depending on a logger that is not logging.
        if self.depends_on is not None and not self.depends_on.log_this_round:
            self.log_this_round = False
            return

        # Log if at the level of the logger.
        if self.trigger_level >= self.logging_level:
            self.log_this_round = True
            return

        # Top-level loggers use can use frequency.
        if self.depends_on is None:
            if self.accumulated_progress * 100 >= self.logging_level:
                self.accumulated_progress = 0
                self.log_this_round = True
                return

        if self.add_first and self.is_first:
            self.log_this_round = True
            return

        if self.add_last and self.is_last:
            self.log_this_round = True
            return

    def log_to_file(self, logs):
        # Make all logs floats to avoid saving large tensors:
        logs = {k: tensor_to_native_type(v) for k, v in logs.items()}
        torch.save(logs, self.log_dir / f"{logs[self.counter_key]}.tar")


def dict_with_prefix(d, prefix):
    """Adds a prefix to all keys in a dict."""
    return {prefix + k: v for k, v in d.items()}


def dict_with_prefixed_match(d, prefix, match):
    """Adds a prefix right before match for all keys containing match in a dict.

    E.g., dict_with_prefixed_match({"a": 1, "b": 2}, "prefix_", "a") == {"prefix_a": 1, "b": 2}
    """
    return {k.replace(match, prefix + match): v for k, v in d.items()}


class DictWithPrefix(dict):
    """A dictionary class that has an update_with_prefix() that appends a prefix the to the keys being updated."""

    def __init__(self, prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    def update_with_prefix(self, new):
        for k, v in new.items():
            self[self.prefix + k] = v


def filter_out_underscore(d):
    """Filters out the keys ending with _"""
    return {k: v for k, v in d.items() if not k.endswith("_")}


def filter_out_wandb(d):
    """Filters out the keys ending with wandb"""
    return {k: v for k, v in d.items() if not k.endswith("wandb")}
