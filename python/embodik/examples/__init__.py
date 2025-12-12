"""Simple examples for embodiK.

This subpackage contains minimal examples that demonstrate basic usage
of embodiK. For more complex examples, see the examples/ directory
in the repository root.
"""

from .basic_ik import basic_ik_example
from .robot_model import robot_model_example

__all__ = [
    "basic_ik_example",
    "robot_model_example",
    "run_example",
]


def run_example(example_name: str):
    """Run a simple example by name.

    Args:
        example_name: Name of the example to run. Options:
            - "basic_ik": Basic inverse kinematics example
            - "robot_model": Robot model loading example

    Examples:
        >>> import embodik.examples
        >>> embodik.examples.run_example("basic_ik")
    """
    if example_name == "basic_ik":
        basic_ik_example()
    elif example_name == "robot_model":
        robot_model_example()
    else:
        raise ValueError(
            f"Unknown example: {example_name}. "
            f"Available examples: basic_ik, robot_model"
        )
