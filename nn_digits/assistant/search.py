"""
A tiny search engine written with NumPy: it finds the text that best answers a question.

It is not a language model: it does not understand the question, it compares words.
  1. Every text becomes a list of "roots" of words: lowercase, without accents, without the words that say
     nothing ("the", "is", "di", "che"...) and without the endings (neurons -> neuron, neuroni -> neuron).
  2. TF-IDF: every text becomes a vector with one number per root. A root counts more if it appears often in
     that text (TF) and less if it appears in many texts (IDF): "loss" says little, "dropout" says a lot.
  3. The question becomes a vector in the same way, and the best text is the one whose vector points in the
     most similar direction (cosine similarity: 1 = same words, 0 = no words in common).
It is the same idea as the "embeddings" of language models, only with counted words instead of learned numbers.
"""
import re
import unicodedata

import numpy as np

# Words that appear in every question and say nothing about the topic (English and Italian, without accents)
STOPWORDS = set("""
a about after again all also am an and any are as at be because been before being but by can could did do does
doing done for from had has have how i if in into is it its just me more most my no not now of on once only or
other our out over own same should so some such than that the their them then there these they this those through
to too under up very was we were what when where which while who why will with would you your yours tell explain
show mean means meaning happen happens thing things get gets got use used using work works one ones please
il lo la i gli le un uno una di da in con su per tra fra a e ed o che chi cui non si mi ti ci vi ne del dello della
dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo
sulla sui sugli sulle col coi cos cosa cose come dove quando perche quale quali quanto quanta quanti quante questo
questa questi queste quello quella quelli quelle sono sei siamo siete era erano essere sta stai sto stanno ho hai
ha abbiamo avete hanno mio mia miei mie tuo tua tuoi tue suo sua suoi sue anche ma se piu molto poi gia ancora
fa fare fai faccio serve servono significa vuol dire spiega spiegami dimmi mostra puoi posso qui qua c l d un po
dell dall nell sull all quest quell cioe ecco uso usa usare funziona funzionano
""".split())


def plain(text):
    """Lowercase and without accents: "Perché è" -> "perche e"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def stem(word):
    """The root of a word, cutting the common endings of English and Italian."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]                       # learning -> learn
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"                 # probabilities -> probability
    if len(word) > 4 and word.endswith("ed") and not word.endswith("eed"):
        return word[:-2]                       # trained -> train
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]                       # neurons -> neuron (but loss stays loss)
    for ending, root in (("che", "c"), ("chi", "c"), ("ghe", "g"), ("ghi", "g")):
        if len(word) > 4 and word.endswith(ending):
            return word[:-len(ending)] + root  # epoche -> epoc, like epoca
    if len(word) > 3 and word[-1] in "aeiou":
        return word[:-1]                       # neurone, neuroni -> neuron;  pesi, peso -> pes
    return word


def words(text):
    """The roots of the words of a text that say something about its topic."""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", plain(text)) if w not in STOPWORDS and len(w) > 1]


class Index:
    """The TF-IDF vectors of some texts, ready to be compared with a question."""

    def __init__(self, texts):
        roots = [words(text) for text in texts]
        self.vocabulary = {w: k for k, w in enumerate(sorted({w for text in roots for w in text}))}
        counts = np.zeros((len(texts), len(self.vocabulary)), np.float32)
        for row, text in enumerate(roots):
            for w in text:
                counts[row, self.vocabulary[w]] += 1
        # IDF: log of (how many texts / in how many texts the root appears), +1 so no root counts zero
        self.idf = np.log((1 + len(texts)) / (1 + (counts > 0).sum(axis=0))) + 1
        self.vectors = self._normalize(np.log1p(counts) * self.idf)  # log1p: 10 repetitions do not count 10 times

    @staticmethod
    def _normalize(vectors):
        """Every vector gets length 1: then the dot product is the cosine of the angle between them."""
        return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)

    def scores(self, question):
        """How much each text resembles the question: one number between 0 and 1 per text."""
        vector = np.zeros(len(self.vocabulary), np.float32)
        for w in words(question):
            if w in self.vocabulary:
                vector[self.vocabulary[w]] += 1
        if not vector.any():
            return np.zeros(len(self.vectors), np.float32)
        return self.vectors @ self._normalize(np.log1p(vector) * self.idf)
