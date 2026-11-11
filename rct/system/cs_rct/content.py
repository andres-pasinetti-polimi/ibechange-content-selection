"""The content rule: which recommendation, which resource and which reminders.

rct/paper/main.tex, Sections "Content rule" and "Fixed rules", and Table "Status of
each delivery element". Every choice is a uniform random draw from a set fixed by
rule, and returns the set, the excluded items with the rule that excluded them, and
the probability of the item drawn, so that content policies can be evaluated on
the logs after the trial.

- Recommendation: the rules of the clinical team (email of 23 October 2026) applied with known
  probabilities (paper, Section "Content rule"). The variations of one recommendation share its
  code. Eligible are the items tagged to the open Mission, less every variation of a code the
  participant disliked and the codes already in the day's plan.
  1. A code not yet sent comes before a variation of a code already sent, always.
  2. Among new codes, the prioritised class (the codes whose COM-B sub-domain is a barrier
     flagged in Step 3, or reflective motivation for pre-intenders and intenders in Smoking and
     Alcohol) is drawn with the priority probability, the other class otherwise; an empty class
     leaves the other with probability 1 and the note records it. Within the class the code is
     drawn uniformly, and within the code the variation.
  3. Once every code has been sent, the code sent least recently is chosen, never one sent the
     day before, and within it a variation never sent, or else the one sent least recently;
     ties are drawn uniformly.
- Resource: one per Mission, in the first plan of the Mission and in place of that day's
  recommendation, drawn among the resources tagged to the Mission that the participant has
  not received; where none is left, the day keeps its recommendation.
- Reminder at the declared hour: on a day whose task is scheduled, the fixed reminder
  is the one written for the Mission's calendar phrase, sent at the declared hour.
- Pop-up text of a recommendation or resource: drawn among the IDs of
  catalogue/push_texts.csv, never the ID of the participant's previous recommendation or resource
  nor one already in the day's plan, as the workbook of the texts asks ("avoid sending the same ID
  twice in a row").
- Day reminders: the fixed reminder of a day without a declared hour and every
  additional reminder are drawn without replacement from the 8 of the Mission's
  pool (reminders of version 3, 27 October 2026), less those sent on the preceding day
  (design.REMINDER_NO_REPEAT_DAYS), so that at least one day passes before a text returns.
"""

from dataclasses import dataclass, field


@dataclass
class Choice:
    content_id: str | None
    probability: float | None
    candidates: list = field(default_factory=list)  # the set drawn from
    excluded: dict = field(default_factory=dict)    # content_id -> rule that excluded it
    note: str | None = None
    rule: dict | None = None  # recommendation only: class, its probability, prioritised items
    # Every random draw (and every choice fixed by rule) that produced content_id, in order, as
    # written to the table draws (schema.sql): {"kind", "options": {option: probability},
    # "chosen", "forced"}. The product of the probabilities of the chosen options is `probability`.
    draws: list = field(default_factory=list)


def record(kind, options, chosen, forced=False):
    """One entry of Choice.draws. Options and the chosen option are written as text, so that they
    can be keys of a JSON object."""
    return {"kind": kind, "options": {str(option): probability for option, probability in options.items()},
            "chosen": None if chosen is None else str(chosen), "forced": forced}


def draw_uniform(rng, candidates, kind=None, draws=None):
    """One item drawn uniformly, and its probability. With `draws`, the draw is appended to it."""
    if not candidates:
        return None, None
    chosen = candidates[int(rng.integers(len(candidates)))]
    if draws is not None:
        draws.append(record(kind, {item: 1 / len(candidates) for item in candidates}, chosen))
    return chosen, 1 / len(candidates)


def choose_recommendation(rng, tagged, disliked, last_sent, code_of=None, prioritised=None,
                          priority_probability=0.0, today_codes=(), yesterday_codes=()):
    """Draw the day's recommendation (rules in the module docstring).

    tagged: the content ids tagged to the Mission, in catalogue order. disliked: ids the
    participant disliked. last_sent: {content_id: last day it was sent}, earlier days only.
    code_of: content id -> code shared by its variations (default: every id its own code).
    prioritised: the set of ids of the prioritised class, or None when the participant has no
    priority (no Step-3 profile, or no barrier flagged). today_codes: codes already in the day's
    plan. yesterday_codes: codes sent the day before.
    """
    code_of = code_of or (lambda item: item)
    disliked_codes = {code_of(item) for item in disliked}
    sent_codes = {code_of(item) for item in last_sent}
    excluded = {}
    eligible = []
    for item in tagged:
        if code_of(item) in disliked_codes:
            excluded[item] = "disliked" if item in disliked else "variation_of_disliked"
        elif code_of(item) in today_codes:
            excluded[item] = "in_todays_plan"
        else:
            eligible.append(item)
    # Not implemented, under consideration (15/10/26): an item that was sent but never
    # opened could count as not sent, because the push body does not reveal the item and
    # the participant has not seen its content. In the pilot 50.4% of the sent
    # recommendations were opened, so this would roughly halve how fast a Mission's pool
    # (median 7 items) is used up. It needs the opening event of each delivery, and it
    # holds only if the item cannot be read elsewhere in the app (library, Mission page)
    # without that event; to be checked with the app team before it is applied.
    new = [item for item in eligible if code_of(item) not in sent_codes]
    if new:
        excluded.update({item: "code_already_sent" for item in eligible if code_of(item) in sent_codes})
        return choose_new(rng, new, excluded, code_of, prioritised, priority_probability)

    candidates = [item for item in eligible if code_of(item) not in yesterday_codes]
    excluded.update({item: "code_sent_yesterday" for item in eligible if code_of(item) in yesterday_codes})
    if not candidates:
        note = ("no_recommendation_tagged" if not tagged else
                "all_disliked" if all(code_of(item) in disliked_codes for item in tagged) else
                "only_codes_sent_yesterday_or_today")
        return Choice(None, None, [], excluded, note)
    return choose_variation(rng, candidates, excluded, code_of, last_sent)


def choose_new(rng, new, excluded, code_of, prioritised, priority_probability):
    """Rule 2: the class, then a code uniformly within it, then a variation uniformly within the code."""
    draws = []
    if prioritised is None:
        chosen_class, class_probability, note = "no_priority", 1.0, "no_priority"
        members = new
    else:
        matched = [item for item in new if item in prioritised]
        unmatched = [item for item in new if item not in prioritised]
        if matched and unmatched:
            note = None
            if rng.random() < priority_probability:
                chosen_class, class_probability, members = "prioritised", priority_probability, matched
            else:
                chosen_class, class_probability, members = "other", 1 - priority_probability, unmatched
        elif matched:
            chosen_class, class_probability, members, note = "prioritised", 1.0, matched, "other_class_empty"
        else:
            chosen_class, class_probability, members, note = "other", 1.0, unmatched, "prioritised_class_empty"
        if note is None:
            draws.append(record("class", {"prioritised": priority_probability, "other": 1 - priority_probability},
                                chosen_class))
    if note is not None:
        draws.append(record("class", {chosen_class: 1.0}, chosen_class, forced=True))
    codes = list(dict.fromkeys(code_of(item) for item in members))
    code, code_probability = draw_uniform(rng, codes, "code", draws)
    variations = [item for item in members if code_of(item) == code]
    content_id, variation_probability = draw_uniform(rng, variations, "variation", draws)
    rule = {"class": chosen_class, "class_probability": class_probability,
            "prioritised": None if prioritised is None else [item for item in new if item in prioritised]}
    return Choice(content_id, class_probability * code_probability * variation_probability, new, excluded,
                  note, rule, draws)


def choose_variation(rng, candidates, excluded, code_of, last_sent):
    """Rule 3: the code sent least recently, and within it a variation never sent or else the one
    sent least recently; ties drawn uniformly."""
    never = "0001-01-01"  # sorts before every day, for a variation never sent

    def last_day(items):
        return max(str(last_sent.get(item, never)) for item in items)

    codes = list(dict.fromkeys(code_of(item) for item in candidates))
    variations_of = {code: [item for item in candidates if code_of(item) == code] for code in codes}
    oldest = min(last_day(variations_of[code]) for code in codes)
    draws = []
    code, code_probability = draw_uniform(rng, [code for code in codes if last_day(variations_of[code]) == oldest],
                                          "code", draws)
    variations = variations_of[code]
    oldest = min(str(last_sent.get(item, never)) for item in variations)
    content_id, variation_probability = draw_uniform(
        rng, [item for item in variations if str(last_sent.get(item, never)) == oldest], "variation", draws)
    rule = {"class": "variation", "class_probability": 1.0, "prioritised": None,
            "code_last_sent": last_day(variations)}
    return Choice(content_id, code_probability * variation_probability, candidates, excluded, "variation", rule,
                  draws)


def choose_resource(rng, tagged, received):
    """Draw a resource among those tagged to the Mission that the participant has not received;
    a resource is never sent twice. Returns a Choice whose content_id is None if none is left."""
    excluded = {item: "already_received" for item in tagged if item in received}
    candidates = [item for item in tagged if item not in received]
    draws = []
    content_id, probability = draw_uniform(rng, candidates, "resource", draws)
    return Choice(content_id, probability, candidates, excluded,
                  None if candidates else ("no_resource_tagged" if not tagged else "all_received"), draws=draws)


def choose_push_text(rng, ids, used_today, previous_id):
    """Draw the ID of the pop-up text of one recommendation or resource.

    ids: the IDs of the texts. used_today: IDs already in the day's plan. previous_id: the ID of the
    participant's last recommendation or resource before today, or None.
    """
    excluded = {item: "used_today" for item in ids if item in used_today}
    if previous_id in ids and previous_id not in excluded:
        excluded[previous_id] = "previous_delivery"
    candidates = [item for item in ids if item not in excluded]
    draws = []
    content_id, probability = draw_uniform(rng, candidates, "push_text", draws)
    return Choice(content_id, probability, candidates, excluded, None if candidates else "no_push_text", draws=draws)


def choose_day_reminders(rng, pool, recently_sent, how_many):
    """Draw how_many day reminders without replacement.

    pool: the reminder ids of the Mission's pool, in catalogue order. recently_sent: ids
    sent within design.REMINDER_NO_REPEAT_DAYS before today. If the pool less the recent ones
    is too small, the recent ones are allowed again (a rule that the 8 texts of a pool make
    unreachable at four reminders a day: four sent the day before leave four).
    """
    choices = []
    chosen_today = []
    for _ in range(how_many):
        excluded = {item: "sent_today" for item in chosen_today}
        candidates = [item for item in pool if item not in chosen_today and item not in recently_sent]
        excluded.update({item: "sent_recently" for item in pool
                         if item in recently_sent and item not in chosen_today})
        note = None
        if not candidates:
            candidates = [item for item in pool if item not in chosen_today]
            excluded = {item: "sent_today" for item in chosen_today}
            note = "repeat_within_no_repeat_days" if candidates else "no_reminder_in_pool"
        draws = []
        content_id, probability = draw_uniform(rng, candidates, "reminder_text", draws)
        choices.append(Choice(content_id, probability, candidates, excluded, note, draws=draws))
        if content_id is not None:
            chosen_today.append(content_id)
    return choices
