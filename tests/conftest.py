"""Runs before every test: the tests always see the program in English, whatever the saved language is."""
import i18n

i18n.use("en")
