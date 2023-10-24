"""ops utils"""
import os
import logging

OPS_NAME = set()
OPS_SCOPE = []

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def check_path_exists(save_path: str) -> None:
    """Check if path exists, if not create it.

    Args:
        save_path (str): Path to check.

    Returns:
        None: Nothing.
    """
    if not os.path.exists(save_path):
        try:
            # Create the directory and any missing parent directories
            os.makedirs(save_path)
            logger.info(f"Directory '{save_path}' created successfully.")
        except OSError as e:
            logger.info(f"Error creating directory '{save_path}': {e}")
    else:
        logger.info(f"Directory '{save_path}' already exists.")
