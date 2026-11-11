# sim_storage.py
import os, json, time
from pathlib import Path
from datetime import datetime
from typing import Any


def _json_default(o: Any):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


class FileDataStorage:
    """
    Append-only JSONL files per run_id.
    Very fast, no DB, easy to load into pandas.
    """

    def __init__(self, out_dir: str | None = None):
        self.out_dir = Path(out_dir or os.getenv("SIM_OUT_DIR", "./sim_out"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = int(time.time() * 1e6)  # unique enough for sim
        self.run_meta_path = self.out_dir / f"run_{self.run_id}_runs.jsonl"
        self.uf_path = self.out_dir / f"run_{self.run_id}_user_feedback.jsonl"
        self.int_path = self.out_dir / f"run_{self.run_id}_intervention_mab_updates.jsonl"
        self.rec_run_path = self.out_dir / f"run_{self.run_id}_recommendation_mab_runs.jsonl"
        self.rec_up_path = self.out_dir / f"run_{self.run_id}_recommendation_mab_updates.jsonl"

        self._buf_user_feedback = []
        self._buf_inter_updates = []
        self._flush_every = int(os.getenv("SIM_FLUSH_EVERY", "5000"))

        # write the run row once
        self._append(
            self.run_meta_path,
            {
                "run_id": self.run_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "env_name": os.getenv("ENV_NAME"),
                "world_id": _safe_int(os.getenv("WORLD_ID") or os.getenv("SCENARIO_ID")),
                "policy_name": os.getenv("POLICY_NAME") or os.getenv("CS_POLICY_NAME"),
                "seed": _safe_int(os.getenv("WORLD_SEED") or os.getenv("SCENARIO_SEED")),
                "algo_seed": _safe_int(os.getenv("ALGO_SEED")),
                "study_tag": os.getenv("STUDY_TAG"),
            },
        )

    def _append(self, path: Path, row: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def _flush(self):
        if self._buf_user_feedback:
            with self.uf_path.open("a", encoding="utf-8") as f:
                for row in self._buf_user_feedback:
                    f.write(json.dumps(row, default=_json_default) + "\n")
            self._buf_user_feedback.clear()

        if self._buf_inter_updates:
            with self.int_path.open("a", encoding="utf-8") as f:
                for row in self._buf_inter_updates:
                    f.write(json.dumps(row, default=_json_default) + "\n")
            self._buf_inter_updates.clear()

    # ---- paper-required writes ----

    def add_user_feedback(self, feedback: dict):
        # feedback shape: {user_id: {"events": [..]}}
        for user_id, payload in (feedback or {}).items():
            for ev in payload.get("events", []):
                self._buf_user_feedback.append(
                    {
                        "run_id": self.run_id,
                        "user_id": user_id,
                        "process_id": ev.get("process_id"),
                        "timestamp": ev.get("timestamp"),
                        "event_name": ev.get("event_name"),
                        "properties": ev.get("properties") or {},
                    }
                )

        if len(self._buf_user_feedback) >= self._flush_every:
            self._flush()

    def add_intervention_mab_update(self, update: dict):
        # store only what param recovery needs; you can add timestamp/id if you want
        self._buf_inter_updates.append(
            {
                "run_id": self.run_id,
                "id": update.get("id"),  # optional
                "timestamp": update.get("timestamp"),
                "params": update.get("params"),
            }
        )
        if len(self._buf_inter_updates) >= self._flush_every:
            self._flush()

    # Optional: regret plots
    def add_recommendation_mab_run(self, bandit_type: str):
        self._append(
            self.rec_run_path,
            {
                "run_id": self.run_id,
                "bandit_type": bandit_type,
                "world_id": _safe_int(os.getenv("WORLD_ID") or os.getenv("SCENARIO_ID")),
            },
        )

    def add_recommendation_mab_update(self, timestamp, reward):
        self._append(self.rec_up_path, {"run_id": self.run_id, "timestamp": timestamp, "reward": reward})

    def finalize_run(self):
        self._flush()


def _safe_int(x):
    try:
        return None if x is None else int(x)
    except Exception:
        return None
