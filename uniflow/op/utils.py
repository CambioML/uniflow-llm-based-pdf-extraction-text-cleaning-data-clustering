"""ops utils"""

OPS_NAME = set()
OPS_SCOPE = []


def clear_ops_name():
    """Clear ops name."""
    OPS_NAME.clear()


def get_op_scope_name(name: str) -> str:
    """Get op scope name.

    Args:
        name (str): Name of the op.

    Returns:
        str: Op scope name.
    """
    return "/".join(OPS_SCOPE + [name])
