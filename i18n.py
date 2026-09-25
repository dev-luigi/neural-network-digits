"""
Languages: the program is written in English, and tr() returns each text in the chosen language.

Translations live in locales/<language>.json as {"English text": "translated text"}.
A text without a translation is shown in English. The language is saved in the settings
(data/settings.json) and is applied at the next start; the first time, the system language is used.
"""
import json
import locale
from pathlib import Path

from neural_net import storage

LANGUAGES = {"en": "English", "it": "Italiano"}
LOCALES_DIR = Path(__file__).resolve().parent / "locales"
LANGUAGE = "en"
_translations = {}
_originals = {}  # translated text -> English text: the assistant searches in both languages


def system_language():
    """Italian if the computer is set to Italian, otherwise English."""
    name = (locale.getlocale()[0] or "").lower()
    return "it" if name.startswith("it") else "en"


def use(language):
    """Switch every following tr() to this language."""
    global LANGUAGE, _translations, _originals
    LANGUAGE = language if language in LANGUAGES else "en"
    file = LOCALES_DIR / f"{LANGUAGE}.json"
    _translations = json.loads(file.read_text(encoding="utf-8")) if LANGUAGE != "en" and file.exists() else {}
    _originals = {translated: english for english, translated in _translations.items()}


def tr(text, **values):
    """The text in the current language, with its {placeholders} filled in: tr("Epoch {n}", n=3)."""
    text = _translations.get(text, text)
    return text.format(**values) if values else text


def english(text):
    """The English original of a text returned by tr() (without {placeholders}); the text itself otherwise."""
    return _originals.get(text, text)


use(storage.settings().get("language") or system_language())
