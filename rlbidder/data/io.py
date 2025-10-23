import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def display_file_summary(directory: Path | str, pattern: str, max_display: int = 5) -> None:
    """Display a summary of files matching a pattern in a directory.
    
    Args:
        directory: Directory to search for files
        pattern: Glob pattern to match files
        max_display: Maximum number of files to display in detail
    """
    directory = Path(directory)
    output_files = sorted(directory.glob(pattern))
    if not output_files:
        logger.warning("No files matching '%s' found in %s", pattern, directory)
        return
    logger.info("Found %d files in %s:", len(output_files), directory)
    for f in output_files[:max_display]:
        logger.info("  - %s", f.name)
    if len(output_files) > max_display:
        logger.info("  ... and %d more files.", len(output_files) - max_display)
