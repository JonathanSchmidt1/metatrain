def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # Add validation_interval (new hyper) and convert log/checkpoint intervals to float
    checkpoint["train_hypers"].setdefault("validation_interval", 1.0)
    checkpoint["train_hypers"]["log_interval"] = float(
        checkpoint["train_hypers"].get("log_interval", 1)
    )
    checkpoint["train_hypers"]["checkpoint_interval"] = float(
        checkpoint["train_hypers"].get("checkpoint_interval", 25)
    )
