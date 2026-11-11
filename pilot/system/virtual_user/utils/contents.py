import json
import os


def load_json_files(type=None):
    folder = "virtual_user/utils/contents"
    filenames = ["missions.json", "recommendations.json", "resources.json"]
    loaded_files = []

    for name in filenames:
        path = os.path.join(folder, name)
        with open(path, "r") as file:
            loaded_files.append(json.load(file))

    if type == "as_dict":
        loaded_files[0] = {c["mission_id"]: c for c in loaded_files[0]}
        loaded_files[1] = {c["rec_id"]: c for c in loaded_files[1]}
        loaded_files[2] = {c["rec_id"]: c for c in loaded_files[2]}

    return tuple(loaded_files)
