"""
``.datasets``
-------------

The ``datasets`` module provides unified access to a number of real-world datasets, including the
General Social Survey, American Life Panel Survey, British Household Panel Survey, and Understanding
Society Survey.
"""
from .general_social_survey import load_general_social_survey  # noqa

__all__ = [
    'load_general_social_survey',
]
