from cs_module.utils.encoding import get_intervention_encoding
from cs_module.config import INTERVENTION_TYPES
import bisect


class RecommendationHistoryTracker:
    """Tracks frequency of recommendations, keeping entries ordered by time."""

    def __init__(self):
        # store (timestamp, process_id, rec_id, mix_vector, mission_id)
        self.history = []

    def add_recommendation(self, timestamp, process_id, rec_id, intervention_type, mission_id, prompted):
        """Add a recommendation (auto-sorted by time)."""
        mix = get_intervention_encoding(intervention_type, keep_all=True)

        row = {
            "timestamp": timestamp,
            "process_id": process_id,
            "rec_id": rec_id,
            "intervention_mix": mix,  # same as before
            "mission_id": mission_id,
            "prompted": prompted,
        }

        # keep list sorted by timestamp
        # we'll bisect on timestamp to preserve chronological order
        idx = bisect.bisect_left([r["timestamp"] for r in self.history], timestamp)
        self.history.insert(idx, row)

    def get_count(self, time_window=None, rec_id=None, exclude_prompted=False):
        start, end = time_window if time_window else (None, None)
        cnt = 0
        for row in self.history:
            if exclude_prompted and row["prompted"]:
                continue
            ts = row["timestamp"]
            if start and ts < start:
                continue
            if end and ts >= end:
                break
            if rec_id is None or row["rec_id"] == rec_id:
                cnt += 1
        return cnt

    def get_type_counters(self, time_window=None):
        counters = [0.0] * len(INTERVENTION_TYPES)
        start, end = time_window if time_window else (None, None)

        for row in self.history:
            ts = row["timestamp"]
            if start and ts < start:
                continue
            if end and ts >= end:
                break  # history is sorted, so we're done

            mix = row["intervention_mix"]
            for j, w in enumerate(mix):
                counters[j] += w

        return counters
