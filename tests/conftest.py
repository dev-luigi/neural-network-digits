"""Runs before every test: the tests always see the program in English, whatever the saved language is."""
from nn_digits import i18n

i18n.use("en")
