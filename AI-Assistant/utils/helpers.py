import os
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_files_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """List all files in a directory with optional extension filtering."""
    path = Path(directory)
    
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    files = []
    for file_path in path.rglob('*'):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(str(file_path))
    
    return files


def format_sources(sources: List[dict]) -> str:
    """Format source information for display."""
    if not sources:
        return "No sources found."
    
    formatted = "\n\nSources:\n"
    for i, source in enumerate(sources, 1):
        formatted += f"\n{i}. "
        if 'source' in source['metadata']:
            formatted += f"File: {Path(source['metadata']['source']).name}\n"
        if 'page' in source['metadata']:
            formatted += f"   Page: {source['metadata']['page']}\n"
        formatted += f"   Preview: {source['content']}\n"
    
    return formatted


def print_divider(char: str = "-", length: int = 50):
    """Print a divider line."""
    print(char * length)