# agent/command_parser.py

import re

def parse_command(user_input: str) -> dict:
    """
    Returns a dictionary describing the action to perform.
    E.g., {'action': 'plot', 'domain': 'time', 'condition': 'KO', 'sensor': 'acc'}
    """
    text = user_input.lower()
    
    result = {
        'action': None,
        'sensor': None,
        'condition': None,
        'domain': None,
        'feature_rank': False,
    }

    if "plot" in text or "show" in text:
        result['action'] = 'plot'
    elif "important" in text or "feature" in text:
        result['action'] = 'feature_analysis'
        result['feature_rank'] = True

    for sensor in ['acc', 'mic', 'mag']:
        if sensor in text:
            result['sensor'] = sensor

    for condition in ['ok', 'ko']:
        if condition in text:
            result['condition'] = condition.upper()

    if "time" in text:
        result['domain'] = 'time'
    elif "frequency" in text or "fft" in text:
        result['domain'] = 'frequency'

    return result
