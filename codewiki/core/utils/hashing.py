"""Hashing utilities for deterministic module content identification.

This module provides hashing functionality to compute stable, reproducible
hashes for content and module hierarchies. Used for caching and change
detection in documentation generation.
"""

import hashlib


class ModuleHasher:
    """Utility class for computing deterministic content and module hashes.

    All methods are static and produce SHA256 hashes. The hashing is designed
    to be deterministic and stable across runs by sorting inputs before hashing.

    Example:
        >>> content_hash = ModuleHasher.compute_content_hash("def foo(): pass")
        >>> module_hash = ModuleHasher.compute_module_hash(
        ...     component_hashes=["abc123", "def456"],
        ...     child_hashes=["child1", "child2"]
        ... )
    """

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute a SHA256 hash of the given content string.

        Args:
            content: The text content to hash. Can be source code,
                documentation, or any other text.

        Returns:
            A hexadecimal SHA256 hash string (64 characters).

        Example:
            >>> ModuleHasher.compute_content_hash("hello world")
            'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
            >>> ModuleHasher.compute_content_hash("")
            'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_module_hash(
        component_hashes: list[str], child_hashes: list[str]
    ) -> str:
        """Compute a deterministic hash for a module based on its components and children.

        This method creates a stable hash by:
        1. Sorting both input lists independently
        2. Concatenating the sorted lists (components first, then children)
        3. Joining all hash strings
        4. Computing SHA256 of the result

        The sorting ensures that the same set of components and children
        always produces the same module hash, regardless of input order.

        Args:
            component_hashes: List of hash strings for the module's direct
                components (functions, classes, etc.)
            child_hashes: List of hash strings for the module's child modules.

        Returns:
            A hexadecimal SHA256 hash string (64 characters).
            Returns the hash of an empty string if both lists are empty.

        Example:
            >>> ModuleHasher.compute_module_hash(["b", "a"], ["d", "c"])
            # Sorts to ["a", "b"] + ["c", "d"], then hashes "abcd"
            >>> ModuleHasher.compute_module_hash([], [])
            'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        """
        # Sort both lists independently for deterministic ordering
        sorted_components = sorted(component_hashes)
        sorted_children = sorted(child_hashes)

        # Merge sorted lists: components first, then children
        all_hashes = sorted_components + sorted_children

        # Concatenate all hash values into a single string
        combined = "".join(all_hashes)

        # Compute SHA256 of the combined string
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
