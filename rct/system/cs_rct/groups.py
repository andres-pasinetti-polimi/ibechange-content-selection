"""Assignment of participants to the learned or the fixed-probability group within the arm.

The evaluation design is an open decision (proposal_rct_module_architecture_2026-09-25.md,
Section 4.2). With design.FIXED_GROUP_SHARE = 0 every participant is in the learned group
and nothing is drawn. With 0.5, participants are assigned 1:1 in permuted blocks of two
within strata of recruitment centre and age band (under 60, 60 or over), so that the two
groups stay balanced in each stratum. Any other share is drawn as a Bernoulli.
"""

from cs_rct import design


def stratum(participant):
    band = "unknown" if participant.age is None else ("60+" if participant.age >= 60 else "<60")
    return f"{participant.recruitment_center or 'unknown'}|{band}"


def assign(rng, groups_in_stratum):
    """The group of a new participant, given the groups already assigned in the stratum, in order."""
    share = design.FIXED_GROUP_SHARE
    if share <= 0:
        return "learned"
    if share >= 1:
        return "fixed"
    if share == 0.5:
        if len(groups_in_stratum) % 2 == 1:
            # Second member of the block: the group the first did not get.
            return "fixed" if groups_in_stratum[-1] == "learned" else "learned"
        return "fixed" if rng.random() < 0.5 else "learned"
    return "fixed" if rng.random() < share else "learned"
