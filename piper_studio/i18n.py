"""Presentation-only localization; does not alter stored observations or commands."""
import json
from pathlib import Path

_language = 'zh'
_english = json.loads(Path(__file__).with_name('translations.en.json').read_text())
_ordered = sorted(_english, key=len, reverse=True)

def set_language(language):
    global _language
    if language not in {'en', 'zh'}:
        raise ValueError('Language must be en or zh')
    _language = language

def t(value):
    if _language != 'en' or not isinstance(value, str):
        return value
    if value in _english:
        return _english[value]
    for phrase in _ordered:
        value = value.replace(phrase, _english[phrase])
    return value
