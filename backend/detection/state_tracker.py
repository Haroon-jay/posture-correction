import numpy as np


class ExerciseStateTracker(dict):
    def __init__(self, config):
        super().__init__(
            {
                "state_seq": [],
                "DISPLAY_TEXT": np.full((4,), False),
                "COUNT_FRAMES": np.zeros((4,), dtype=np.int64),
                "INCORRECT_POSTURE": False,
                "prev_state": None,
                "curr_state": None,
                "REP_COUNT": 0,
                "IMPROPER_REP_COUNT": 0,
                "state_seq_2": [],
                "exercise": config.name,
            }
        )

    def reset(self):
        self.clear()
        self.update(
            {
                "state_seq": [],
                "DISPLAY_TEXT": np.full((4,), False),
                "COUNT_FRAMES": np.zeros((4,), dtype=np.int64),
                "INCORRECT_POSTURE": False,
                "prev_state": None,
                "curr_state": None,
                "REP_COUNT": 0,
                "IMPROPER_REP_COUNT": 0,
                "state_seq_2": [],
                "exercise": self["exercise"],  # Preserve exercise name
            }
        )

    def to_dict(self):
        """Convert object to JSON-serializable dictionary"""
        return {
            "exercise": self["exercise"],
            "state_seq": self["state_seq"],
            "DISPLAY_TEXT": self[
                "DISPLAY_TEXT"
            ].tolist(),  # Convert numpy array to list
            "COUNT_FRAMES": self[
                "COUNT_FRAMES"
            ].tolist(),  # Convert numpy array to list
            "INCORRECT_POSTURE": self["INCORRECT_POSTURE"],
            "prev_state": self["prev_state"],
            "curr_state": self["curr_state"],
            "REP_COUNT": self["REP_COUNT"],
            "IMPROPER_REP_COUNT": self["IMPROPER_REP_COUNT"],
            "state_seq_2": self["state_seq_2"],
        }
