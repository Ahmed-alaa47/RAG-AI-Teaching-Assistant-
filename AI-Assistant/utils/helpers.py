from pathlib import Path
from typing import List
import logging

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
logger = logging.getLogger(__name__)

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
        # formatted += f"   Preview: {source['content']}\n"
    
    return formatted


def print_divider(char: str = "-", length: int = 50):
    """Print a divider line."""
    print(char * length)