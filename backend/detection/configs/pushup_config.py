import mediapipe as mp
import cv2
import numpy as np
from utils import find_angle, get_complete_coords, draw_dotted_line
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


import cv2


def on_side_view_pushup(coords, angles, COLORS, frame, linetype, font):
    # Safely extract coordinates, handling None values
    def safe_tuple(coord):
        return tuple(map(int, coord[:2])) if coord is not None else None

    shldr_coord = safe_tuple(coords.get("shldr_coord"))
    elbow_coord = safe_tuple(coords.get("elbow_coord"))
    wrist_coord = safe_tuple(coords.get("wrist_coord"))
    hip_coord = safe_tuple(coords.get("hip_coord"))
    ankle_coord = safe_tuple(coords.get("ankle_coord"))  # May be None

    multiplier = coords.get("multiplier", 1)

    # ------------------- Push-Up Angles -------------------
    def draw_angle_marker(coord, angle, color):
        if coord and angle is not None:
            cv2.ellipse(
                frame,
                coord,
                (30, 30),
                angle=0,
                startAngle=-90,
                endAngle=-90 + multiplier * angle,
                color=color,
                thickness=3,
                lineType=linetype,
            )

            draw_dotted_line(
                frame,
                coord,
                start=int(coord[1] - 50),
                end=int(coord[1] + 20),
                line_color=COLORS["blue"],
            )

    draw_angle_marker(elbow_coord, angles.get("elbow_angle"), COLORS["white"])
    draw_angle_marker(shldr_coord, angles.get("shoulder_angle"), COLORS["white"])
    draw_angle_marker(hip_coord, angles.get("hip_angle"), COLORS["white"])

    # ------------------- Connect Body Joints -------------------
    def draw_line(coord1, coord2):
        if coord1 and coord2:
            cv2.line(frame, coord1, coord2, COLORS["light_blue"], 4, lineType=linetype)

    draw_line(shldr_coord, elbow_coord)
    draw_line(elbow_coord, wrist_coord)
    draw_line(shldr_coord, hip_coord)
    if ankle_coord:  # Only draw if ankle exists
        draw_line(hip_coord, ankle_coord)

    # ------------------- Plot Key Landmarks -------------------
    def draw_point(coord):
        if coord:
            cv2.circle(frame, coord, 7, COLORS["yellow"], -1, lineType=linetype)

    draw_point(shldr_coord)
    draw_point(elbow_coord)
    draw_point(wrist_coord)
    draw_point(hip_coord)
    draw_point(ankle_coord)  # Safe to handle None

    # ------------------- Display Angle Values -------------------
    def draw_text(coord, angle):
        if coord and angle is not None:
            cv2.putText(
                frame,
                str(int(angle)),
                (int(coord[0] + 10), int(coord[1])),
                font,
                0.6,
                COLORS["light_green"],
                2,
                lineType=linetype,
            )

    draw_text(elbow_coord, angles.get("elbow_angle"))
    draw_text(shldr_coord, angles.get("shoulder_angle"))
    draw_text(hip_coord, angles.get("hip_angle"))

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
    name="push-up",
)
