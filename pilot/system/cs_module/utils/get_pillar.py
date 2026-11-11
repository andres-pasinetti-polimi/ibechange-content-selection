def get_pillar(id):
    if id.startswith("A"):
        return "alcohol"
    elif id.startswith("N"):
        return "nutrition"
    elif id.startswith("P"):
        return "physical_activity"
    elif id.startswith("S"):
        return "smoking"
    elif id.startswith("E"):
        return "emotional_wellbeing"
    else:
        raise ValueError(f"Unknown pillar for {id}")