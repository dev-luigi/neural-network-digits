"""Translations: every text passed to tr() has its Italian version in locales/it.json, and none is left over."""
import ast
import json
import re
from pathlib import Path

from project import NAME

ROOT = Path(__file__).resolve().parent.parent
ITALIAN = json.loads((ROOT / "locales" / "it.json").read_text(encoding="utf-8"))
SKIP = {"tests", "data", ".git", ".venv", "venv", "env"}


def translated_texts():
    """The English texts the program translates, and the tr() calls that do not get a plain string."""
    texts, not_literal = set(), []
    for path in sorted(ROOT.rglob("*.py")):
        where = path.relative_to(ROOT)
        if SKIP & set(where.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Tiles translates by itself the names it receives, so its names are texts to translate too
        inside_tiles = {id(node) for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) and cls.name == "Tiles"
                        for node in ast.walk(cls)}
        constants = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if function == "tr" and node.args and id(node) not in inside_tiles:
                text = node.args[0]
                if isinstance(text, ast.Name) and text.id == "NAME":  # the program name, from project.py
                    texts.add(NAME)
                elif isinstance(text, ast.Constant) and isinstance(text.value, str):
                    texts.add(text.value)
                else:
                    not_literal.append(f"{where}:{node.lineno}")
            elif function == "Tiles":
                names = node.args[1] if len(node.args) > 1 else next(
                    (k.value for k in node.keywords if k.arg == "names"), None)
                if isinstance(names, ast.Name):  # a constant of the module, like DASHBOARD = ("loss", ...)
                    names = constants.get(names.id)
                texts.update(e.value for e in getattr(names, "elts", []) if isinstance(e, ast.Constant))
    return texts, not_literal


TEXTS, NOT_LITERAL = translated_texts()


def placeholders(text):
    """The names in {braces}: "{acc:.1%} of {total}" -> {"acc", "total"}."""
    return set(re.findall(r"(?<!\{)\{(\w+)[^{}]*\}", text))


def test_tr_gets_only_plain_strings():
    """tr("text"), never tr(f"...") or tr(variable): the English text is the key of its translation."""
    assert NOT_LITERAL == []


def test_every_text_has_its_italian_translation():
    assert sorted(TEXTS - set(ITALIAN)) == []


def test_translations_have_the_same_placeholders():
    assert [key for key, value in ITALIAN.items() if placeholders(key) != placeholders(value)] == []


def test_no_unused_translations():
    assert sorted(set(ITALIAN) - TEXTS) == []
