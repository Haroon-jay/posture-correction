import mediapipe as mp
import cv2
import numpy as np
from utils import (
    find_angle,
    find_angle_3d,
    draw_text,
    get_complete_coords,
    draw_dotted_line,
)
from thresholds import get_thresholds_beginner

from configs.exercise_config import ExerciseConfig

feedback_map = {
    0: ("BEND BACKWARDS", 215, (0, 153, 255)),
    1: ("BEND FORWARD", 215, (0, 153, 255)),
    2: ("KNEE FALLING OVER TOE", 170, (255, 80, 80)),
    3: ("SQUAT TOO DEEP", 125, (255, 80, 80)),
}


def update_state_sequence_squat(state, state_tracker):
    if state == "s2":
        if (
            ("s3" not in state_tracker["state_seq"])
            and (state_tracker["state_seq"].count("s2")) == 0
        ) or (
            ("s3" in state_tracker["state_seq"])
            and (state_tracker["state_seq"].count("s2") == 1)
        ):
            state_tracker["state_seq"].append(state)

    elif state == "s3":
        if (state not in state_tracker["state_seq"]) and "s2" in state_tracker[
            "state_seq"
        ]:
            state_tracker["state_seq"].append(state)


def calc_squat_coords(lm, frame_width, frame_height):
    coords = get_complete_coords(lm, frame_width, frame_height)
    if (
        coords["shldr_coord"] is None
        or coords["elbow_coord"] is None
        or coords["wrist_coord"] is None
        or coords["hip_coord"] is None
        or coords["knee_coord"] is None
        or coords["ankle_coord"] is None
        or coords["foot_coord"] is None
    ):
        return False
    return coords


def on_side_view(coords, angles, COLORS, frame, linetype, font):
    # Extract coordinates and convert to (x, y) tuples
    shldr_coord = tuple(map(int, coords["shldr_coord"][:2]))
    elbow_coord = tuple(map(int, coords["elbow_coord"][:2]))
    wrist_coord = tuple(map(int, coords["wrist_coord"][:2]))
    hip_coord = tuple(map(int, coords["hip_coord"][:2]))
    knee_coord = tuple(map(int, coords["knee_coord"][:2]))
    ankle_coord = tuple(map(int, coords["ankle_coord"][:2]))
    foot_coord = tuple(map(int, coords["foot_coord"][:2]))
    multiplier = coords["multiplier"]

    # ------------------- Vertical Angle Calculation --------------
    hip_vertical_angle = angles["hip_vertical_angle"]
    if hip_vertical_angle is not None:
        cv2.ellipse(
            frame,
            hip_coord,
            (30, 30),
            angle=0,
            startAngle=-90,
            endAngle=-90 + multiplier * hip_vertical_angle,
            color=COLORS["white"],
            thickness=3,
            lineType=linetype,
        )

        draw_dotted_line(
            frame,
            hip_coord,
            start=int(hip_coord[1] - 80),
            end=int(hip_coord[1] + 20),
            line_color=COLORS["blue"],
        )

    knee_vertical_angle = angles["knee_vertical_angle"]
    if knee_vertical_angle is not None:
        cv2.ellipse(
            frame,
            knee_coord,
            (20, 20),
            angle=0,
            startAngle=-90,
            endAngle=-90 - multiplier * knee_vertical_angle,
            color=COLORS["white"],
            thickness=3,
            lineType=linetype,
        )

        draw_dotted_line(
            frame,
            knee_coord,
            start=int(knee_coord[1] - 50),
            end=int(knee_coord[1] + 20),
            line_color=COLORS["blue"],
        )

    ankle_vertical_angle = angles["ankle_vertical_angle"]
    if ankle_vertical_angle is not None:
        cv2.ellipse(
            frame,
            ankle_coord,
            (30, 30),
            angle=0,
            startAngle=-90,
            endAngle=-90 + multiplier * ankle_vertical_angle,
            color=COLORS["white"],
            thickness=3,
            lineType=linetype,
        )

        draw_dotted_line(
            frame,
            ankle_coord,
            start=int(ankle_coord[1] - 50),
            end=int(ankle_coord[1] + 20),
            line_color=COLORS["blue"],
        )

    # Join landmarks with lines
    cv2.line(
        frame, shldr_coord, elbow_coord, COLORS["light_blue"], 4, lineType=linetype
    )
    cv2.line(
        frame, wrist_coord, elbow_coord, COLORS["light_blue"], 4, lineType=linetype
    )
    cv2.line(frame, shldr_coord, hip_coord, COLORS["light_blue"], 4, lineType=linetype)
    cv2.line(frame, knee_coord, hip_coord, COLORS["light_blue"], 4, lineType=linetype)
    cv2.line(frame, ankle_coord, knee_coord, COLORS["light_blue"], 4, lineType=linetype)
    cv2.line(frame, ankle_coord, foot_coord, COLORS["light_blue"], 4, lineType=linetype)

    # Plot landmark points
    cv2.circle(frame, shldr_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, elbow_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, wrist_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, hip_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, knee_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, ankle_coord, 7, COLORS["yellow"], -1, lineType=linetype)
    cv2.circle(frame, foot_coord, 7, COLORS["yellow"], -1, lineType=linetype)

    # Text labels for angles
    hip_text_coord_x = int(hip_coord[0] + 10)
    knee_text_coord_x = int(knee_coord[0] + 15)
    ankle_text_coord_x = int(ankle_coord[0] + 10)

    cv2.putText(
        frame,
        str(int(hip_vertical_angle)) if hip_vertical_angle is not None else "N/A",
        (hip_text_coord_x, int(hip_coord[1])),
        font,
        0.6,
        COLORS["light_green"],
        2,
        lineType=linetype,
    )
    cv2.putText(
        frame,
        str(int(knee_vertical_angle)) if knee_vertical_angle is not None else "N/A",
        (knee_text_coord_x, int(knee_coord[1]) + 10),
        font,
        0.6,
        COLORS["light_green"],
        2,
        lineType=linetype,
    )
    cv2.putText(
        frame,
        str(int(ankle_vertical_angle)) if ankle_vertical_angle is not None else "N/A",
        (ankle_text_coord_x, int(ankle_coord[1])),
        font,
        0.6,
        COLORS["light_green"],
        2,
        lineType=linetype,
    )

    return True


def perform_action_for_state(current_state, state_tracker, angles, thresholds):
    # print(current_state, state_tracker['state_seq'])
    if current_state == "s1":

        if (
            len(state_tracker["state_seq"]) == 3
            and not state_tracker["INCORRECT_POSTURE"]
        ):
            state_tracker["REP_COUNT"] += 1

        # elif 's2' in state_tracker['state_seq'] and len(state_tracker['state_seq'])==1:
        #     state_tracker['IMPROPER_REP_COUNT']+=1

        elif state_tracker["INCORRECT_POSTURE"]:
            state_tracker["IMPROPER_REP_COUNT"] += 1

        state_tracker["state_seq"] = []
        state_tracker["INCORRECT_POSTURE"] = False

    else:
        if angles["hip_vertical_angle"] > thresholds["HIP_THRESH"][1]:
            state_tracker["DISPLAY_TEXT"][0] = True

        elif (
            angles["hip_vertical_angle"] < thresholds["HIP_THRESH"][0]
            and state_tracker["state_seq"].count("s2") == 1
        ):
            state_tracker["DISPLAY_TEXT"][1] = True

        # if thresholds['KNEE_THRESH'][0] < angles["knee_vertical_angle"] < thresholds['KNEE_THRESH'][1] and \
        #     state_tracker['state_seq'].count('s2')==1:
        #     state_tracker['LOWER_HIPS'] = True

        elif angles["knee_vertical_angle"] > thresholds["KNEE_THRESH"][2]:
            state_tracker["DISPLAY_TEXT"][3] = True
            state_tracker["INCORRECT_POSTURE"] = True
            print("squat too deep", state_tracker["state_seq"])

        if angles["ankle_vertical_angle"] > thresholds["ANKLE_THRESH"]:
            state_tracker["DISPLAY_TEXT"][2] = True
            state_tracker["INCORRECT_POSTURE"] = True
            print("knee falling over toe", state_tracker["state_seq"])


def get_squat_state(angles, thresholds, isFront=True):
    knee = None
    knee_angle = int(angles["knee_vertical_angle"])
    if (
        thresholds["HIP_KNEE_VERT"]["NORMAL"][0]
        <= knee_angle
        <= thresholds["HIP_KNEE_VERT"]["NORMAL"][1]
    ):
        knee = 1
    elif (
        thresholds["HIP_KNEE_VERT"]["TRANS"][0]
        <= knee_angle
        <= thresholds["HIP_KNEE_VERT"]["TRANS"][1]
    ):
        knee = 2
    elif (
        thresholds["HIP_KNEE_VERT"]["PASS"][0]
        <= knee_angle
        <= thresholds["HIP_KNEE_VERT"]["PASS"][1]
    ):
        knee = 3

    return f"s{knee}" if knee else None


def calc_squat_angles(coords):

    offset_angle = find_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    hip_vertical_angle = find_angle(
        coords["shldr_coord"],
        np.array([coords["hip_coord"][0], 0]),
        coords["hip_coord"],
    )
    knee_vertical_angle = find_angle(
        coords["hip_coord"],
        np.array([coords["knee_coord"][0], 0]),
        coords["knee_coord"],
    )
    ankle_vertical_angle = find_angle(
        coords["knee_coord"],
        np.array([coords["ankle_coord"][0], 0]),
        coords["ankle_coord"],
    )

    return {
        "offset_angle": offset_angle,
        "hip_vertical_angle": hip_vertical_angle,
        "knee_vertical_angle": knee_vertical_angle,
        "ankle_vertical_angle": ankle_vertical_angle,
    }


def get_thresholds_beginner():

    _ANGLE_HIP_KNEE_VERT = {"NORMAL": (0, 32), "TRANS": (35, 65), "PASS": (70, 95)}

    thresholds = {
        "HIP_KNEE_VERT": _ANGLE_HIP_KNEE_VERT,
        "HIP_THRESH": [10, 50],
        "ANKLE_THRESH": 45,
        "KNEE_THRESH": [50, 70, 95],
        "OFFSET_THRESH": 35.0,
        # 'INACTIVE_THRESH'  : 15.0,
        "CNT_FRAME_THRESH": 50,
    }

    return thresholds


def on_front_view(
    coords, angles, state_tracker, frame, frame_width, frame_height, COLORS
):
    print("left_shldr_coord:", coords["left_shldr_coord"])
    print("Type of left_shldr_coord:", type(coords["left_shldr_coord"]))
    print("First two elements:", coords["left_shldr_coord"][:2])

    cv2.circle(
        frame, tuple(map(int, coords["left_shldr_coord"][:2])), 7, COLORS["yellow"], -1
    )
    cv2.circle(frame, tuple(map(int, coords["nose_coord"][:2])), 7, COLORS["white"], -1)
    cv2.circle(
        frame,
        tuple(map(int, coords["right_shldr_coord"][:2])),
        7,
        COLORS["magenta"],
        -1,
    )

    # cv2.circle(frame, coords["left_shldr_coord"][:2], 7, COLORS["yellow"], -1)
    # cv2.circle(frame, coords["nose_coord"][:2], 7, COLORS["white"], -1)
    # cv2.circle(frame, coords["right_shldr_coord"][:2], 7, COLORS["magenta"], -1)

    draw_text(
        frame,
        "CAMERA NOT ALIGNED PROPERLY!!!",
        pos=(30, frame_height - 60),
        text_color=(255, 255, 230),
        font_scale=0.65,
        text_color_bg=(255, 153, 0),
    )

    draw_text(
        frame,
        "OFFSET ANGLE: " + str(angles["offset_angle"]),
        pos=(30, frame_height - 30),
        text_color=(255, 255, 230),
        font_scale=0.65,
        text_color_bg=(255, 153, 0),
    )

    state_tracker["prev_state"] = None
    state_tracker["curr_state"] = None
    return False


squat_config = ExerciseConfig(
    coords_calc_fn=calc_squat_coords,
    angles_calc_fn=calc_squat_angles,
    thresholds_fn=get_thresholds_beginner,
    get_state_fn=get_squat_state,
    perform_action_fn=perform_action_for_state,
    view_fns={"front": on_front_view, "side": on_side_view},
    feedback_map={
        0: ("BEND BACKWARDS", 215, (0, 153, 255)),
        1: ("BEND FORWARD", 215, (0, 153, 255)),
        2: ("KNEE FALLING OVER TOE", 170, (255, 80, 80)),
        3: ("SQUAT TOO DEEP", 125, (255, 80, 80)),
    },
    update_state_sequence=update_state_sequence_squat,
    name="squat",
)
