"""
CodeWiki - A tool for generating comprehensive documentation from Python codebases.

This module orchestrates the documentation generation process by:
1. Analyzing code dependencies
2. Clustering related modules
3. Generating documentation using AI agents
4. Creating overview documentation
"""

import logging
import argparse
import asyncio
import os

# Configure logging and monitoring
from codewiki.src.be.dependency_analyzer.utils.logging_config import setup_logging

# Initialize colored logging
setup_logging(level=logging.INFO)

logger = logging.getLogger(__name__)

# Local imports
from codewiki.src.be.documentation_generator import DocumentationGenerator
from codewiki.core import Settings


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive documentation for Python components in dependency order.'
    )
    parser.add_argument(
        '--repo-path',
        type=str,
        required=True,
        help='Path to the repository'
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point for the documentation generation process."""
    try:
        # Parse arguments and create configuration
        args = parse_arguments()

        # Create Settings directly from environment
        settings = Settings(
            repo_path=args.repo_path,
            # Other settings loaded from env vars via CODEWIKI_ prefix
        )

        # Prepare output directories
        repo_name = os.path.basename(os.path.normpath(args.repo_path))
        sanitized_repo_name = ''.join(c if c.isalnum() else '_' for c in repo_name)
        output_dir = os.path.join("output", "docs", f"{sanitized_repo_name}-docs")
        dependency_graph_dir = os.path.join("output", "dependency_graphs")

        # Create and run documentation generator
        doc_generator = DocumentationGenerator(
            settings=settings,
            repo_path=args.repo_path,
            output_dir=output_dir,
            dependency_graph_dir=dependency_graph_dir
        )
        await doc_generator.run()

    except KeyboardInterrupt:
        logger.debug("Documentation generation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())