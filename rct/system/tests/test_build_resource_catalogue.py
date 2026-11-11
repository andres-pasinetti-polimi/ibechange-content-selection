from scripts.build_resource_catalogue import expand

KNOWN = {"SM1", "SM2", "SM3", "PM1", "PM2"}
OF_PILLAR = {"Smoking": ["SM1", "SM2", "SM3"], "Physical_activity": ["PM1", "PM2"]}


def test_all_gives_the_missions_of_the_pillar():
    assert expand("All", "Smoking", OF_PILLAR, KNOWN) == (["SM1", "SM2", "SM3"], [])


def test_ranges_with_and_without_padding_and_spaces():
    assert expand("SM01 - SM02", "Smoking", OF_PILLAR, KNOWN) == (["SM1", "SM2"], [])
    assert expand("PM1-PM2", "Physical_activity", OF_PILLAR, KNOWN) == (["PM1", "PM2"], [])


def test_lists_and_codes_outside_the_catalogue():
    assert expand("SM3, SM1-SM2, SM9", "Smoking", OF_PILLAR, KNOWN) == (["SM3", "SM1", "SM2"], ["SM9"])


def test_empty_cell_gives_no_missions():
    assert expand(None, "Smoking", OF_PILLAR, KNOWN) == ([], [])
