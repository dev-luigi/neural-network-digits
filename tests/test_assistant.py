"""The assistant: the small search engine, the answers (in English and in Italian) and the hints.
The state of the program is a dictionary written by hand, like the one made by gui/app_state.py."""
import numpy as np
import pytest

import i18n
from assistant import knowledge, rules
from assistant.brain import Assistant, facts, next_text, status_text
from assistant.search import Index, plain, stem, words

PARAMS = dict(lr=0.05, momentum=0.9, l2=1e-4, dropout=0.0, batch=32, noise=0.0, rotation=12, shift=2)
CHOSEN = {"hidden": [128, 64], "activation": "relu", "init_scale": 1.0, "seed": 0}


def state(tab="training", history=None, photos=5000, model=True, evaluation=None, lab=None, **params):
    return {"tab": tab, "photos": {"train": photos, "test": photos // 5}, "model": model,
            "training": {"running": False, "paused": False, "params": dict(PARAMS, **params), "cosine": True,
                         "chosen": dict(CHOSEN), "network": {"layers": [784, 128, 64, 10], "activation": "relu",
                                                             "init_scale": 1.0},
                         "history": history},
            "evaluation": evaluation or {"noise": 0.0, "rotation": 0, "thickness": 0, "threshold": 0.0,
                                         "accuracy": 0.9, "total": 1000},
            "lab": lab, "drawing": None, "inside": None}


def history(val_acc, train_acc=None, val_loss=None, train_loss=None, inactive=(0.0, 0.0)):
    """The measurements of some epochs: by default a network that learns well."""
    n = len(val_acc)
    return {"val_acc": list(val_acc), "train_acc": list(val_acc if train_acc is None else train_acc),
            "val_loss": list(np.linspace(1, 0.3, n) if val_loss is None else val_loss),
            "train_loss": list(np.linspace(1, 0.25, n) if train_loss is None else train_loss),
            "lr": [0.05] * n, "inactive": [list(inactive)] * n}


LEARNING_WELL = history([0.80, 0.86, 0.88, 0.90, 0.91, 0.915, 0.92, 0.925])


@pytest.fixture
def italian():
    i18n.use("it")
    yield
    i18n.use("en")


def keys(s):
    return [hint.key for hint in rules.hints(s)]


# ------------------------------------------------------------------ the search engine

def test_words_become_roots_in_both_languages():
    assert stem("neurons") == stem("neuroni") == stem("neurone") == "neuron"
    assert stem("epoche") == stem("epoca")
    assert stem("loss") == "loss" and stem("learning") == "learn"
    assert plain("Perché È") == "perche e"
    assert words("What is the Loss?") == ["loss"]


def test_index_finds_the_most_similar_text():
    index = Index(["the dropout switches off neurons", "the learning rate is the size of the step",
                   "the photos of the digits"])
    scores = index.scores("what does dropout do?")
    assert scores.argmax() == 0 and scores[0] > 0.3
    assert index.scores("banana").max() == 0


# ------------------------------------------------------------------ the answers

@pytest.mark.parametrize("question, title", [
    ("what is the learning rate?", "Learning rate"),
    ("what is a neuron?", "Neuron, weights and bias"),
    ("what is overfitting?", "Overfitting"),
    ("what is t-SNE?", "Point map (PCA and t-SNE)"),
    ("what does the momentum do", "Momentum"),
    ("how does backpropagation work", "Backpropagation and gradient descent"),
    ("difference between validation and test", "Train, validation and test"),
    ("what is my accuracy", "Accuracy"),
    ("how is it going?", "How it is going"),
    ("what should I do now?", "What to do now"),
    ("is something wrong?", "What I see"),
    ("what can you do?", "The assistant"),
    ("hi", "Hi!"),
    ("what is this tab", "Tab 2 · Training"),
    ("how do I change the language?", "Language of the program"),  # not "Temperature" (of language models)
])
def test_answers(question, title):
    assert Assistant().answer(question, state(history=LEARNING_WELL)).title == title


@pytest.mark.parametrize("question, title", [
    ("cos'è un neurone?", "Neurone, pesi e bias"),
    ("a cosa serve la temperatura", "Temperatura"),
    ("cosa sono i neuroni inattivi", "Neuroni inattivi"),
    ("cos'è il learning rate", "Velocità di apprendimento"),
    ("cos'è l'epoca", "Epoca"),
    ("la rete è esplosa", "Rete esplosa"),
    ("come sta andando?", "Come sta andando"),
    ("cosa faccio adesso", "Cosa fare adesso"),
    ("perché non impara?", "Cosa vedo"),
    ("ciao", "Ciao!"),
    ("come cambio la lingua?", "Lingua del programma"),
    ("what is a neuron?", "Neurone, pesi e bias"),  # in English too, whatever the language in use
])
def test_answers_in_italian(italian, question, title):
    assert Assistant().answer(question, state(history=LEARNING_WELL)).title == title


def test_problems_are_counted():
    assistant = Assistant()
    fine = assistant.answer("is something wrong?", state(history=LEARNING_WELL))
    assert fine.text == "I see nothing wrong right now."
    assert assistant.answer("is something wrong?", state(noise=0.5)).text == "I see one thing to fix."
    reply = assistant.answer("is something wrong?", state(noise=0.5, rotation=40))
    assert reply.text.startswith("I see 2 things") and [h.key for h in reply.hints] == ["noise", "rotation"]


def test_nonsense_gets_an_honest_answer():
    reply = Assistant().answer("asdfgh qwerty", state())
    assert not reply.found and reply.chips


@pytest.mark.parametrize("language", ["en", "it"])
def test_every_suggested_question_has_an_answer(language):
    i18n.use(language)
    try:
        assistant = Assistant()
        for tab in knowledge.TABS:
            for question in knowledge.suggestions(tab):
                assert assistant.answer(question, state(tab, history=LEARNING_WELL)).found, question
    finally:
        i18n.use("en")


def test_the_controls_can_be_searched():
    assistant = Assistant([("Maximum rotation", "Data augmentation: every photo is rotated randomly.", "training")])
    found = assistant.search("maximum rotation")
    assert found[0][1].kind == "control" and found[0][1].title == "Maximum rotation"


def test_a_reply_says_what_the_concept_is_worth_now_and_where_it_is():
    reply = Assistant().answer("what is the learning rate?", state("evaluation", history=LEARNING_WELL))
    notes = dict(reply.notes)
    assert notes["Right now"] == "the learning rate is 0.05, with cosine decay (in the last epoch: 0.05)."
    assert notes["Where"] == "tab 2 · Training" and "Try it" in notes


def test_hints_follow_the_topic_or_all_of_them_for_my_network():
    exploded = history([0.5, 0.3, 0.1], train_loss=[1.0, 5.0, np.nan])
    assistant = Assistant()
    assert [h.key for h in assistant.answer("what is dropout?", state(history=exploded, lr=1.0)).hints] == []
    assert [h.key for h in assistant.answer("tell me about my dropout", state(history=exploded, lr=1.0)).hints] \
        == ["exploded"]


def test_status_and_next_step_follow_the_program():
    assert "no photos" in status_text(state(photos=0))
    assert "Download the photos" in next_text(state(photos=0))
    assert "Start" in status_text(state())
    assert "8 epochs done" in status_text(state(history=LEARNING_WELL))
    assert "no trained model" in status_text(state("evaluation", model=False))
    assert "90.0% of the 1000" in status_text(state("evaluation"))


def test_facts():
    s = state(history=LEARNING_WELL)
    assert facts("dropout", s) == "dropout 0%."
    assert facts("batch", s) == "32 photos per mini-batch."
    assert facts("seed", s).startswith("in tab 2: layers 784 → 128 → 64 → 10, activation relu")
    assert facts("epoch", s).startswith("epoch 8: train accuracy 92.5%")
    assert facts("confusion", s) is None


# ------------------------------------------------------------------ the hints

def test_no_hints_when_everything_is_fine():
    assert keys(state(history=LEARNING_WELL)) == []
    assert keys(state()) == []  # a new network, not trained yet


def test_every_hint_explains_a_real_concept():
    concepts = {c.key for c in knowledge.concepts()}
    exploded = history([0.5, 0.3, 0.1], train_loss=[1.0, 5.0, np.nan])
    for s in (state(photos=0), state("evaluation", model=False), state(history=exploded, lr=1.0),
              state(history=history([0.1] * 4)), state(tab="draw", lab={"accuracy_before": 0.9,
                                                                          "accuracy_after": 0.5})):
        for hint in rules.hints(s):
            assert hint.topic in concepts and hint.text


def test_missing_photos_and_model():
    assert keys(state(photos=0)) == ["photos"]
    assert keys(state("data", photos=0)) == []  # in tab 1 you are already where you download them
    assert keys(state("evaluation", model=False)) == ["model"]


def test_exploded_network():
    s = state(history=history([0.5, 0.3, 0.1], train_loss=[1.0, 5.0, np.nan]), lr=1.0)
    [hint] = rules.hints(s)
    assert hint.key == "exploded" and hint.fix == (("set", "lr", 0.05), ("new network",))


def test_not_learning_finds_the_reason():
    stuck = history([0.1, 0.11, 0.1, 0.09])
    assert rules.hints(state(history=stuck, lr=0.8))[0].fix[0] == ("set", "lr", 0.05)
    assert rules.hints(state(history=stuck, lr=0.0001))[0].fix == (("set", "lr", 0.05),)
    s = state(history=stuck)
    s["training"]["network"]["init_scale"] = 0.1
    assert rules.hints(s)[0].fix[0] == ("set", "init_scale", 1.0)
    assert keys(state(history=stuck)) == ["not learning"]


def test_slow_learning():
    assert keys(state(history=history([0.3, 0.4, 0.5, 0.55, 0.6]), lr=0.001)) == ["slow"]


def test_jumping_loss():
    jumping = history([0.9] * 6, val_loss=[0.5, 0.4, 0.6, 0.35, 0.7, 0.4])
    assert keys(state(history=jumping, lr=0.5)) == ["jumping"]
    assert keys(state(history=jumping, lr=0.05)) == []


def test_overfitting_and_its_fixes():
    val_loss = [0.6, 0.4, 0.3, 0.31, 0.33, 0.35, 0.38, 0.40]
    train_loss = [0.6, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
    rising = history([0.9] * 8, val_loss=val_loss, train_loss=train_loss)
    assert rules.hints(state(history=rising))[0].fix == (("set", "dropout", 0.2),)
    assert rules.hints(state(history=rising, dropout=0.2))[0].fix == (("set", "l2", 1e-3),)
    assert rules.hints(state(history=rising, dropout=0.2, l2=1e-3, rotation=0))[0].fix[0] == ("set", "rotation", 12)
    assert rules.hints(state(history=rising, dropout=0.2, l2=1e-3))[0].fix == (("tab", "data"),)
    gap = history([0.8] * 6, train_acc=[0.99] * 6)
    assert keys(state(history=gap)) == ["overfitting"]


def test_inactive_neurons():
    assert keys(state(history=history([0.85] * 3, inactive=(0.5, 0.1)))) == ["inactive"]
    assert "layer 1" in rules.hints(state(history=history([0.85] * 3, inactive=(0.5, 0.1))))[0].text


def test_extreme_controls():
    s = state(history=LEARNING_WELL)
    s["training"]["chosen"]["init_scale"] = 5.0
    assert keys(s) == ["initial weights"]
    assert keys(state(history=history([0.5] * 3), l2=0.05)) == ["l2"]
    assert keys(state(history=history([0.5] * 3), dropout=0.7)) == ["dropout"]
    assert keys(state(noise=0.5)) == ["noise"]
    assert keys(state(rotation=40)) == ["rotation"]
    assert keys(state(momentum=0.99, lr=0.2)) == ["momentum"]
    assert keys(state(batch=2)) == ["batch"]
    assert keys(state(photos=100)) == ["few photos"]
    assert keys(state("evaluation", noise=0.5)) == []  # the extreme controls of tab 2 matter only there


def test_test_photos_harder_than_training():
    rotated = {"noise": 0.0, "rotation": 30, "thickness": 0, "threshold": 0.0, "accuracy": 0.6, "total": 1000}
    [hint] = rules.hints(state("evaluation", evaluation=rotated))
    assert hint.fix == (("set", "rotation", 30), ("tab", "training"))
    noisy = dict(rotated, rotation=0, noise=0.4)
    assert keys(state("evaluation", evaluation=noisy)) == ["test noise"]
    assert keys(state("evaluation", evaluation=noisy, noise=0.2)) == []


def test_lab_damage():
    damaged = {"changed": True, "accuracy_before": 0.9, "accuracy_after": 0.5}
    assert keys(state("draw", lab=damaged)) == ["lab"]
    assert keys(state("draw", lab=dict(damaged, accuracy_after=0.85))) == []


def test_a_smaller_learning_rate():
    for lr in (1.0, 0.5, 0.3, 0.1):
        assert lr / 4 < rules._smaller_lr(lr) < lr / 2
