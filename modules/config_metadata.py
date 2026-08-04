CONFIG_TEMPLATE_METADATA_KEYS = frozenset({"_comment"})


def split_config_template_metadata(loaded_config):
    """Separate reserved template metadata from runtime configuration values."""
    if not isinstance(loaded_config, dict):
        raise ValueError("The configuration root must be a JSON object.")

    runtime_config = dict(loaded_config)
    metadata = {}

    if "_comment" in runtime_config:
        comment = runtime_config.pop("_comment")
        is_valid_comment = isinstance(comment, str) or (
            isinstance(comment, list)
            and all(isinstance(line, str) for line in comment)
        )
        if not is_valid_comment:
            raise ValueError(
                'Reserved config metadata "_comment" must be a string or a list of strings.'
            )
        metadata["_comment"] = comment

    return runtime_config, metadata
