import mediapipe as mp
import cv2
import numpy as np
from utils import find_angle, get_complete_coords
from thresholds import get_thresholds_pushup

from configs.exercise_config import ExerciseConfig


def get_pushup_state(angles, thresholds, isFront=True):
    elbow = None
    try:
        elbow_angle = int(angles["elbow_angle"])
        key = "FRONT" if isFront else "SIDE"
        print(key)
        if (
            thresholds["ELBOW_ANGLE"][key]["START"][0]
            <= elbow_angle
            <= thresholds["ELBOW_ANGLE"][key]["START"][1]
        ):
            elbow = 1
        elif (
            thresholds["ELBOW_ANGLE"][key]["TRANS"][0]
            <= elbow_angle
            <= thresholds["ELBOW_ANGLE"][key]["TRANS"][1]
        ):
            elbow = 2
        elif thresholds["ELBOW_ANGLE"][key]["COMPLETE"] <= elbow_angle:
            elbow = 3

        return f"s{elbow}" if elbow else None
    except:
        return None


def perform_action_for_state_pushup(current_state, state_tracker, angles, thresholds):
    if current_state == "s1":

        if len(state_tracker["state_seq"]) == 2:
            state_tracker["IMPROPER_REP_COUNT"] += 1
            state_tracker["DISPLAY_TEXT"][0] = True
            state_tracker["state_seq"] = []
            state_tracker["INCORRECT_POSTURE"] = False

        if (
            len(state_tracker["state_seq"]) == 4
            and not state_tracker["INCORRECT_POSTURE"]
        ):
            state_tracker["REP_COUNT"] += 1
            state_tracker["state_seq"] = []
            state_tracker["INCORRECT_POSTURE"] = False

        # elif 's2' in state_tracker['state_seq'] and len(state_tracker['state_seq'])==1:
        #     state_tracker['IMPROPER_REP_COUNT']+=1

        elif state_tracker["INCORRECT_POSTURE"]:
            state_tracker["IMPROPER_REP_COUNT"] += 1
            state_tracker["INCORRECT_POSTURE"] = False


def on_front_view_pushup(
    coords, angles, state_tracker, frame, frame_width, frame_height, COLORS
):
    return False


def calc_pushup_coords(lm, frame_width, frame_height):
    coords = get_complete_coords(lm, frame_width, frame_height)
    if (
        coords["shldr_coord"] is None
        or coords["elbow_coord"] is None
        or ["wrist_coord"] is None
        or ["hip_coord"] is None
        or ["knee_coord"] is None
    ):
        if ["shldr_coord"] is None:
            print("Shoulder not visible")
        if ["elbow_coord"] is None:
            print("Elbow not visible")
        if ["wrist_coord"] is None:
            print("Wrist not visible")
        if ["hip_coord"] is None:
            print("Hip not visible")
        if ["knee_coord"] is None:
            print("Knee not visible")

        return False
    return coords


def calc_pushup_angles(coords):

    offset_angle = find_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    # Elbow angle: upper arm (shoulder to elbow) and forearm (elbow to wrist)
    elbow_angle = find_angle(
        coords["shldr_coord"], coords["elbow_coord"], coords["wrist_coord"]
    )

    # Body line: shoulder to hip to knee
    body_line_angle = find_angle(
        coords["shldr_coord"], coords["hip_coord"], coords["knee_coord"]
    )

    # Vertical alignment of wrist and shoulder
    wrist_shldr_alignment = find_angle(
        coords["wrist_coord"],
        np.array([coords["shldr_coord"][0], 0]),
        coords["shldr_coord"],
    )

    return {
        "elbow_angle": elbow_angle,
        "body_line_angle": body_line_angle,
        "wrist_shldr_alignment": wrist_shldr_alignment,
        "offset_angle": offset_angle,
    }


def on_side_view_pushup(coords, angles, COLORS, frame, linetype, font):
    return True


def update_state_sequence_pushup(state, state_tracker):
    if state == "s1":
        if len(state_tracker["state_seq"]) == 0:
            state_tracker["state_seq"].append(state)
    if state == "s2":
        if (
            ("s3" not in state_tracker["state_seq"])
            and (state_tracker["state_seq"].count("s2")) == 0
            and (state_tracker["state_seq"].count("s1") == 1)
        ) or (
            ("s3" in state_tracker["state_seq"])
            and (state_tracker["state_seq"].count("s2") == 1)
            and (state_tracker["state_seq"].count("s1") == 1)
        ):
            state_tracker["state_seq"].append(state)

    elif state == "s3":
        if (state not in state_tracker["state_seq"]) and "s2" in state_tracker[
            "state_seq"
        ]:
            state_tracker["state_seq"].append(state)


pushup_config = ExerciseConfig(
    coords_calc_fn=calc_pushup_coords,
    angles_calc_fn=calc_pushup_angles,
    thresholds_fn=get_thresholds_pushup,
    get_state_fn=get_pushup_state,
    perform_action_fn=perform_action_for_state_pushup,
    view_fns={"front": on_front_view_pushup, "side": on_side_view_pushup},
    feedback_map={
        0: ("KEEP GOING", 215, (0, 153, 255)),
        1: ("INCOMPLETE PUSHUP", 215, (0, 153, 255)),
    },
    update_state_sequence=update_state_sequence_pushup,
)
