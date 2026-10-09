"""Logging utilities for the AITW monomer infiltration surrogate project."""
import logging

from rich.logging import RichHandler


def get_logger(name: str, output_file: str = None) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name (str): The name of the logger.
        output_file (str): The path to the file where the log messages will be saved.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a RichHandler for pretty logging output
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    console_formatter = logging.Formatter(
        "{asctime} - {name} - {levelname:>7s} - {message}", style="{",
    )
    rich_handler.setFormatter(console_formatter)

    file_handler = None
    if output_file:
        # Create a file handler for logging to a file
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the file handler
        file_formatter = logging.Formatter(
            "{asctime} - {levelname:>7s} - {message}", style="{"
        )
        file_handler.setFormatter(file_formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(rich_handler)
        if file_handler:
            logger.addHandler(file_handler)

    return logger

def set_console_level(logger: logging.Logger, level: int) -> None:
    """Set the logging level for the console handler of the logger.

    Args:
        logger (logging.Logger): The logger whose console handler's level is to be set.
        level (int): The logging level to set for the console handler.
    """
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level)
