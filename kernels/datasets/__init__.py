"""
This module provides unified access to a number of real-world datasets, including the
General Social Survey, American Life Panel Survey, British Household Panel Survey, and Understanding
Society Survey.
"""
from .american_life_panel import AmericanLifePanelDataset  # noqa
from .general_social_survey import GeneralSocialSurveyDataset  # noqa
from .understanding_society import UnderstandingSocietyDataset  # noqa

__all__ = [
    'AmericanLifePanelDataset',
    'GeneralSocialSurveyDataset',
    'UnderstandingSocietyDataset',
]
