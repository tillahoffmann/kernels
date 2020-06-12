"""
This module provides unified access to a number of real-world datasets, including the
General Social Survey, American Life Panel Survey, British Household Panel Survey, and Understanding
Society Survey.
"""
from .american_life_panel import load_american_life_panel  # noqa
from .general_social_survey import load_general_social_survey  # noqa
from .understanding_society import load_understanding_society_survey  # noqa

__all__ = [
    'load_american_life_panel',
    'load_general_social_survey',
    'load_understanding_society_survey',
]
