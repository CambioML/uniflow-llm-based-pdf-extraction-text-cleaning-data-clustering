import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

QA_FACTOR = 3


def set_qa_factor(factor):
    """Set QA Factor

    Args:
        factor (int): Number of additional QA pairs to generate for each QA pair.

    Returns:
        None
    """
    global QA_FACTOR
    logger.debug(f"Setting QA_FACTOR: {factor}")
    QA_FACTOR = factor


def get_qa_factor():
    logger.debug(f"Getting QA_FACTOR: {QA_FACTOR}")
    return QA_FACTOR
