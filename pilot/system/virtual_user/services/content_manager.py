from virtual_user.utils.contents import load_json_files
from cs_module.utils.get_pillar import get_pillar


class ContentManager:
    def __init__(self):
        self.missions, self.recommendations, self.resources = load_json_files("as_dict")

    def get_available_recommendations_and_resources(self, user_new_missions, unavailable_resources):
        available_recommendations = {}
        available_resources = {}

        for mission_id in user_new_missions:
            pillar = get_pillar(mission_id)
            available_recommendations[mission_id] = []
            available_resources[mission_id] = []

            for rec_id, rec in self.recommendations.items():
                if get_pillar(rec_id) == pillar and mission_id in rec["mission"]:
                    available_recommendations[mission_id].append(rec_id)

            for res_id, res in self.resources.items():
                if get_pillar(res_id) == pillar and res_id not in unavailable_resources:
                    if res["mission"]:
                        for res_mission_id in res["mission"]:
                            if res_mission_id in user_new_missions:
                                available_resources[res_mission_id].append(res_id)

        return available_recommendations, available_resources
