"""
How the assistant answers a question.

  1. It recognizes a few kinds of questions about the program itself ("how is it going?", "what should I do
     now?", "is something wrong?", "what is this tab?"): for these it reads the real state of the program.
  2. Otherwise it looks for the most similar text among the concepts of knowledge.py, the explanations of the
     controls and the descriptions of the tabs (search.py). To the found text it adds what that concept is worth
     right now in the program, where to see it and an experiment to try.
  3. If nothing looks similar enough, it says so honestly and suggests some questions.

Every text is searched both in English and in the language in use: you can ask in both languages.
"""
import re
from collections import namedtuple

import numpy as np

from assistant import knowledge, rules
from assistant.search import Index, plain, words
from i18n import english, tr
from project import VERSION

# kind: "concept", "control" or "tab"; key: the concept (or the tab) it talks about
Document = namedtuple("Document", "kind key title text tab experiment")
# notes: (label, text) lines under the text; hints: rules.Hint to show with their fix; chips: questions to suggest
Reply = namedtuple("Reply", "title text notes hints chips found")

MINIMUM_SCORE = 0.12  # below this the texts are too different from the question
STRONG_SCORE = 0.35   # above this a found text wins even over a question about the program

# Ways of asking the questions about the program (English and Italian, lowercase and without accents)
GREETINGS = ("hello", "hi", "hey", "ciao", "salve", "buongiorno", "buonasera")
HELP = ("help", "aiuto", "what can you do", "cosa sai fare", "che sai fare", "how do you work", "come funzioni",
        "who are you", "what are you", "chi sei", "cosa sei", "come ti uso", "how do i use you")
TAB = ("this tab", "this page", "what can i do here", "what am i looking at", "what do i see", "where am i",
       "questa scheda", "questa pagina", "questa tab", "cosa posso fare qui", "cosa vedo", "dove sono")
NEXT = ("what should i do", "what do i do", "what now", "now what", "what next", "next step", "how do i start",
        "where do i start", "what can i do", "what should i try", "cosa faccio", "cosa devo fare", "che faccio",
        "che devo fare", "prossimo passo", "e adesso", "e ora", "come inizio", "da dove inizio", "da dove parto",
        "cosa posso fare", "cosa provo", "cosa posso provare")
PROBLEMS = ("what s wrong", "what is wrong", "something wrong", "anything wrong", "is it wrong", "wrong", "problem",
            "problems", "mistake", "hint", "hints", "advice", "tip", "tips", "suggestion", "suggestions", "improve",
            "not learning", "doesn t learn", "does not learn", "not working", "doesn t work", "stuck",
            "cosa non va", "qualcosa non va", "non va", "cosa sbaglio", "ho sbagliato", "sbagliato", "sbaglio",
            "problema", "problemi", "errore", "suggerimento", "suggerimenti", "consiglio", "consigli", "migliorare",
            "non impara", "non funziona", "bloccata", "bloccato")
STATUS = ("how is it going", "how s it going", "how is the training", "how am i doing", "how did it go", "how good",
          "status", "progress", "results", "summary", "come va", "come sta andando", "come procede", "a che punto",
          "com e andata", "stato", "andamento", "risultati", "riassunto", "situazione", "quanto e brava",
          "quanto e bravo")
DEFINITION = ("what is", "what are", "what does", "what do", "explain", "mean", "meaning", "define", "difference",
              "why", "how does", "cos e", "cosa e", "cosa sono", "spiega", "spiegami", "significa", "vuol dire",
              "a cosa serve", "a che serve", "differenza", "perche", "come funziona")
MINE = ("my", "mine", "mio", "mia", "miei", "mie")


def _contains(question, phrases):
    """Is one of the phrases in the question, as whole words?"""
    return any(f" {phrase} " in question for phrase in phrases)


class Assistant:
    def __init__(self, controls=()):
        """controls = (name, explanation, tab) of the controls of the interface: they can be searched too."""
        self.documents = [Document("concept", c.key, c.title, c.text, c.tab, c.experiment)
                          for c in knowledge.concepts()]
        self.documents += [Document("tab", tab, knowledge.tab_name(tab), knowledge.tab_description(tab), tab, "")
                           for tab in knowledge.TABS]
        # A control with the same name as a concept (Learning rate, Momentum...): the concept explains it better
        concept_titles = {plain(d.title) for d in self.documents}
        self.documents += [Document("control", "", name, text, tab, "") for name, text, tab in controls
                           if plain(name) not in concept_titles]
        keywords = {c.key: c.keywords for c in knowledge.concepts()}
        self.index = Index([self._searchable(d, keywords.get(d.key, "")) for d in self.documents])
        self.titles = [set(words(f"{d.title} {english(d.title)}")) for d in self.documents]

    @staticmethod
    def _searchable(document, keywords):
        """The words to search: the title and the keywords count more than the text, in both languages."""
        parts = [document.title] * 3 + [keywords] * 2 + [document.text]
        return " ".join(parts + [english(part) for part in parts])

    # ------------------------------------------------------------------ answering

    def search(self, question, tab=None):
        """The documents most similar to the question: [(score, document)], best first."""
        scores = self.index.scores(question)
        asked = set(words(question))
        for k, document in enumerate(self.documents):
            if asked and scores[k] > 0:  # the words of the question in the title: it is about exactly that
                scores[k] += 0.1 * len(asked & self.titles[k]) / len(asked)
            if document.kind == "concept":
                scores[k] *= 1.15  # a concept explains better than the explanation of a single control
            if document.tab == tab:
                scores[k] *= 1.1   # what is in the open tab is more likely what you are asking about
        order = np.argsort(-scores)
        return [(float(scores[k]), self.documents[k]) for k in order if scores[k] >= MINIMUM_SCORE]

    def answer(self, question, state):
        """The reply to a question, looking at the state of the program (see gui/app_state.py)."""
        text = " " + " ".join(re.findall(r"[a-z0-9]+", plain(question))) + " "
        tab = state.get("tab", "info")
        if not text.strip():
            return self.fallback(tab)
        if _contains(text, GREETINGS) and len(text.split()) <= 3:
            return self.hello(state)
        if _contains(text, HELP):
            return self.document_reply(self.concept("assistant"), state)

        intent = next((reply for phrases, reply in ((TAB, self.this_tab), (NEXT, self.next_step),
                                                     (PROBLEMS, self.problems), (STATUS, self.status))
                       if _contains(text, phrases)), None)
        found = self.search(question, tab)
        if intent and not (found and found[0][0] >= STRONG_SCORE and _contains(text, DEFINITION)):
            return intent(state)
        if found:
            return self.document_reply(found[0][1], state, related=found[1:], mine=_contains(text, MINE))
        return self.fallback(tab)

    def concept(self, key):
        """The document of a concept of knowledge.py."""
        return next(d for d in self.documents if d.kind == "concept" and d.key == key)

    def document_reply(self, document, state, related=(), mine=False, value=None):
        """A found text, with what it is worth now, where to see it and something to try."""
        notes = []
        if value:
            notes.append((tr("Right now"), value))
        elif document.kind == "concept":
            fact = facts(document.key, state)
            if fact:
                notes.append((tr("Right now"), fact))
        if document.kind != "tab" and document.tab != state.get("tab") and document.key != "assistant":
            notes.append((tr("Where"), tr("tab {tab}", tab=knowledge.tab_name(document.tab))))
        if document.kind == "tab":
            notes.append((tr("Try it"), knowledge.experiment(document.key)))
        elif document.experiment:
            notes.append((tr("Try it"), document.experiment))
        active = rules.hints(state)
        if not mine:  # only the hints about this topic; asking about "my network" shows all of them
            active = [hint for hint in active if hint.topic == document.key]
        best = related[0][0] if related else 0
        chips = []  # the related texts to click, each title once (two tabs can have controls with the same name)
        for score, d in related[:4]:
            if score >= 0.6 * best and d.title and d.title != document.title and d.title not in chips:
                chips.append(d.title)
        return Reply(document.title, document.text, notes, active, chips[:2], True)

    def hello(self, state):
        return Reply(tr("Hi!"), tr("I am the assistant of the program. Ask me about the network, the controls or what "
                                   "you see, or ask \"how is it going?\" and \"what should I do now?\". With Pick (F1) "
                                   "click a control and I explain it."), [], [], [], True)

    def fallback(self, tab):
        return Reply(tr("I did not understand"), tr("I am not a language model: I look for the words of your question "
                                                    "in my texts, and I found nothing similar. Try with other words, "
                                                    "or with one of these questions."),
                     [], [], knowledge.suggestions(tab), False)

    def this_tab(self, state):
        tab = state.get("tab", "info")
        return Reply(tr("Tab {tab}", tab=knowledge.tab_name(tab)), knowledge.tab_description(tab),
                     [(tr("Right now"), status_text(state)), (tr("Try it"), knowledge.experiment(tab))],
                     rules.hints(state), [], True)

    def status(self, state):
        return Reply(tr("How it is going"), status_text(state), [], rules.hints(state), [], True)

    def problems(self, state):
        active = rules.hints(state)
        if len(active) == 1:
            return Reply(tr("What I see"), tr("I see one thing to fix."), [], active, [], True)
        if active:
            text = tr("I see {n} things to fix, the most important first.", n=len(active))
            return Reply(tr("What I see"), text, [], active, [], True)
        return Reply(tr("What I see"), tr("I see nothing wrong right now."), [(tr("Next step"), next_text(state))], [],
                     [], True)

    def next_step(self, state):
        return Reply(tr("What to do now"), next_text(state), [], rules.hints(state)[:2], [], True)


# ------------------------------------------------------------------ the state in words

def status_text(state):
    """What is happening in the open tab, in a few words."""
    tab, training = state.get("tab", "info"), state.get("training") or {}
    photos = state.get("photos", {})
    if tab == "info":
        return tr("You are using version {version}.", version=VERSION)
    if tab == "data":
        if not photos.get("train"):
            return tr("There are no photos yet: press Download the photos.")
        return tr("There are {train} training photos and {test} test photos.", train=photos["train"],
                  test=photos.get("test", 0))
    if not photos.get("train"):
        return tr("There are no photos yet: download them in tab 1 · Data.")
    if tab == "training":
        return _training_status(training)
    if not state.get("model"):
        return tr("There is no trained model yet: train the network in tab 2 · Training.")
    if tab == "evaluation":
        return _evaluation_status(state.get("evaluation") or {})
    if tab == "draw":
        return _draw_status(state.get("drawing"), state.get("lab"))
    inside = state.get("inside") or {}
    if "photo" not in inside:
        return tr("Here you see the math of the saved network.")
    return tr("Photo no. {photo}, {layer}: the true digit is {true}, the network answers {answer}.", **inside)


def _training_status(training):
    history = training.get("history")
    if not history or not history.get("val_acc"):
        return tr("A new network with random weights is ready: press Start.")
    epoch = len(history["val_acc"])
    if not np.isfinite(history["train_loss"][-1]):
        return tr("The network exploded at epoch {epoch}: the weights have become infinite.", epoch=epoch)
    if training.get("running"):
        text = (tr("Training paused at epoch {epoch}.", epoch=epoch) if training.get("paused")
                else tr("Training in progress: epoch {epoch}.", epoch=epoch))
    else:
        text = tr("{epoch} epochs done.", epoch=epoch)
    val_loss = history["val_loss"]
    best = int(np.argmin(val_loss))
    text += " " + tr("Accuracy: {train:.1%} on the training photos, {val:.1%} on the validation ones.",
                     train=history["train_acc"][-1], val=history["val_acc"][-1])
    if best == len(val_loss) - 1:
        text += " " + tr("The validation loss is still going down ({loss:.3f}).", loss=val_loss[-1])
    else:
        text += " " + tr("The validation loss has not improved for {n} epochs (the best was {loss:.3f} at epoch "
                         "{epoch}).", n=len(val_loss) - 1 - best, loss=val_loss[best], epoch=best + 1)
    return text


def _evaluation_status(evaluation):
    if evaluation.get("accuracy") is None:
        return tr("The evaluation is not ready yet.")
    text = tr("The network recognizes {accuracy:.1%} of the {total} test photos.", **evaluation)
    if evaluation.get("noise") or evaluation.get("rotation") or evaluation.get("thickness"):
        text += " " + tr("The photos are spoiled: noise σ {noise:.2f}, rotation {rotation:+.0f}°, thickness "
                         "{thickness:+d}.", noise=evaluation.get("noise", 0), rotation=evaluation.get("rotation", 0),
                         thickness=int(evaluation.get("thickness", 0)))
    return text


def _draw_status(drawing, lab):
    if drawing:
        text = tr("On your drawing the network answers {answer}, {confidence:.0%} sure.", **drawing)
    else:
        text = tr("The board is empty: draw a digit.")
    if lab and lab.get("accuracy_before") is not None and lab.get("changed"):
        text += " " + tr("With your changes the test accuracy goes from {before:.1%} to {after:.1%}.",
                         before=lab["accuracy_before"], after=lab["accuracy_after"])
    return text


def next_text(state):
    """The next step: the first thing missing, or an experiment to try in the open tab."""
    tab, training = state.get("tab", "info"), state.get("training") or {}
    if not state.get("photos", {}).get("train"):
        return tr("Go to tab 1 · Data and press Download the photos (MNIST, about 11 MB, only the first time).")
    if training.get("running"):
        return tr("The network is learning: watch the loss go down and the accuracy go up. The Optimization knobs "
                  "can be changed even now.")
    if not state.get("model"):
        return tr("Go to tab 2 · Training and press Start: with the starting settings it takes less than half a "
                  "minute.")
    if tab == "training":
        return tr("The model is saved: see how it does on photos it has never seen in tab 3 · Evaluation.") + " " + (
            knowledge.experiment(tab))
    return knowledge.experiment(tab)


def facts(key, state):
    """What a concept is worth right now in the program (None if there is nothing to say)."""
    training = state.get("training") or {}
    params, history = training.get("params") or {}, training.get("history") or {}
    chosen, evaluation = training.get("chosen") or {}, state.get("evaluation") or {}
    epochs = len(history.get("val_acc", []))
    if key == "learning rate" and params:
        text = tr("the learning rate is {lr:g}", lr=params["lr"])
        if training.get("cosine"):
            text += tr(", with cosine decay")
        if epochs:
            text += tr(" (in the last epoch: {lr:.2g})", lr=history["lr"][-1])
        return text + "."
    if key in ("momentum", "l2", "dropout", "batch") and params:
        return {"momentum": tr("momentum {value:.2f}.", value=params["momentum"]),
                "l2": tr("L2 {value:g}.", value=params["l2"]),
                "dropout": tr("dropout {value:.0%}.", value=params["dropout"]),
                "batch": tr("{value} photos per mini-batch.", value=params["batch"])}[key]
    if key == "augmentation" and params:
        return tr("rotation up to ±{rotation:.0f}°, shift up to ±{shift} px.", rotation=params["rotation"],
                  shift=params["shift"])
    if key == "noise" and params:
        return tr("noise σ {noise:.2f} on the training photos, {test:.2f} on the test photos.", noise=params["noise"],
                  test=evaluation.get("noise", 0))
    if key in ("accuracy", "overfitting", "epoch", "loss") and epochs:
        return tr("epoch {epoch}: train accuracy {train:.1%}, validation {val:.1%}; train loss {train_loss:.3f}, "
                  "validation {val_loss:.3f}.", epoch=epochs, train=history["train_acc"][-1],
                  val=history["val_acc"][-1], train_loss=history["train_loss"][-1], val_loss=history["val_loss"][-1])
    if key == "inactive" and history.get("inactive"):
        return tr("inactive neurons per layer: {shares}.",
                  shares=", ".join(f"{share:.0%}" for share in history["inactive"][-1]))
    if key in ("initialization", "activation", "seed", "network") and chosen:
        return tr("in tab 2: layers {layers}, activation {activation}, width of the initial weights x{scale:g}, seed "
                  "{seed}.", layers=" → ".join(map(str, [784, *chosen["hidden"], 10])),
                  activation=chosen["activation"], scale=chosen["init_scale"], seed=chosen["seed"])
    if key == "threshold" and "threshold" in evaluation:
        return tr("threshold {threshold:.0%} in tab 3.", threshold=evaluation["threshold"])
    if key in ("mnist", "validation") and state.get("photos", {}).get("train"):
        return tr("{train} training photos and {test} test photos.", train=state["photos"]["train"],
                  test=state["photos"].get("test", 0))
    return None
