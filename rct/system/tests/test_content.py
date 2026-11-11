import numpy as np

from cs_rct import content


def rng():
    return np.random.default_rng(0)


from datetime import date

DAY1, DAY2, DAY3 = date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)
CODE = {"A1a": "A1", "A1b": "A1", "A2": "A2", "B1a": "B1", "B1b": "B1", "B1c": "B1", "B2": "B2"}


def test_recommendation_prefers_codes_not_yet_sent_and_skips_disliked():
    choice = content.choose_recommendation(rng(), ["A", "B", "C", "D"], disliked={"B"}, last_sent={"A": DAY1})
    assert choice.candidates == ["C", "D"]
    assert choice.probability == 0.5 and choice.rule["class"] == "no_priority"
    assert choice.excluded == {"B": "disliked", "A": "code_already_sent"}


def test_a_variation_of_a_sent_code_waits_until_every_code_is_sent():
    choice = content.choose_recommendation(rng(), list(CODE), set(), {"A1a": DAY1}, CODE.get)
    assert choice.excluded == {"A1a": "code_already_sent", "A1b": "code_already_sent"}
    assert CODE[choice.content_id] in ("A2", "B1", "B2")


def test_a_disliked_item_excludes_its_variations():
    choice = content.choose_recommendation(rng(), list(CODE), {"B1a"}, {}, CODE.get)
    assert choice.excluded == {"B1a": "disliked", "B1b": "variation_of_disliked", "B1c": "variation_of_disliked"}


def test_prioritised_class_probability_and_logged_propensity():
    tagged = list(CODE)
    prioritised = {"B1a", "B1b", "B1c", "B2"}
    picks = [content.choose_recommendation(np.random.default_rng(seed), tagged, set(), {}, CODE.get,
                                           prioritised, 0.7) for seed in range(2000)]
    share = sum(p.rule["class"] == "prioritised" for p in picks) / len(picks)
    assert 0.66 < share < 0.74
    for p in picks:
        if p.content_id == "B1c":   # class 0.7, code 1/2 (B1, B2), variation 1/3
            assert abs(p.probability - 0.7 / 2 / 3) < 1e-12
        if p.content_id == "A2":    # class 0.3, code 1/2 (A1, A2), one variation
            assert abs(p.probability - 0.3 / 2) < 1e-12
    assert picks[0].rule["prioritised"] == ["B1a", "B1b", "B1c", "B2"]


def test_an_empty_class_leaves_the_other_with_probability_one():
    choice = content.choose_recommendation(rng(), ["A2"], set(), {}, CODE.get, {"B2"}, 0.7)
    assert choice.content_id == "A2" and choice.probability == 1.0 and choice.note == "prioritised_class_empty"


def test_once_every_code_is_sent_the_least_recently_sent_code_and_an_unsent_variation():
    last_sent = {"A1a": DAY1, "A2": DAY2, "B1a": DAY1, "B2": DAY3}
    tagged = ["A1a", "A1b", "A2", "B1a", "B1b", "B2"]
    code = {"A1a": "A1", "A1b": "A1", "A2": "A2", "B1a": "B1", "B1b": "B1", "B2": "B2"}
    choice = content.choose_recommendation(rng(), tagged, set(), last_sent, code.get,
                                           yesterday_codes={"B2"})
    # A1 and B1 were both last sent on DAY1: one of the two, then its variation never sent.
    assert choice.content_id in ("A1b", "B1b") and choice.probability == 0.5
    assert choice.note == "variation" and choice.excluded == {"B2": "code_sent_yesterday"}


def test_no_recommendation_when_everything_is_disliked():
    choice = content.choose_recommendation(rng(), ["A"], disliked={"A"}, last_sent={})
    assert choice.content_id is None and choice.note == "all_disliked"


def test_day_reminders_are_distinct_and_avoid_the_last_days():
    pool = [f"D{i}" for i in range(14)]
    recent = {f"D{i}" for i in range(9)}
    choices = content.choose_day_reminders(rng(), pool, recent, 3)
    chosen = [c.content_id for c in choices]
    assert len(set(chosen)) == 3 and not set(chosen) & recent
    assert [c.probability for c in choices] == [1 / 5, 1 / 4, 1 / 3]


def test_day_reminders_repeat_only_when_the_pool_is_exhausted():
    choices = content.choose_day_reminders(rng(), ["D1", "D2"], {"D1", "D2"}, 1)
    assert choices[0].note == "repeat_within_no_repeat_days" and choices[0].content_id in ("D1", "D2")
