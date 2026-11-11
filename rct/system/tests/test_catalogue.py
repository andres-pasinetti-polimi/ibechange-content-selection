from cs_rct.catalogue import by_mission, mission_code


def test_mission_code_removes_leading_zeros_and_spaces():
    assert mission_code("SM01") == "SM1"
    assert mission_code(" SM1 ") == "SM1"
    assert mission_code("PM93") == "PM93"
    assert mission_code("EM10") == "EM10"


def test_mission_code_leaves_other_values_unchanged():
    assert mission_code("All") == "All"
    assert mission_code("SM01 - SM49") == "SM01 - SM49"


def test_by_mission_joins_padded_and_unpadded_codes():
    items = {"SRs15": {"missions": ["SM01", "SM2"]}, "SRs16": {"missions": ["SM1", "SM01"]}}
    assert by_mission(items) == {"SM1": ["SRs15", "SRs16"], "SM2": ["SRs15"]}


def test_mission_code_reads_a_space_between_letters_and_number():
    assert mission_code("EM 31") == "EM31"
