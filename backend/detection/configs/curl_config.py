import mediapipe as mp
import cv2
import numpy as np
from utils import (
    draw_text,
    find_angle,
    find_angle_3d,
    get_complete_coords,
    draw_dotted_line,
)
from thresholds import get_thresholds_curl

from configs.exercise_config import ExerciseConfig


def get_curl_state(angles, thresholds, isFront=True):
    elbow = None
    elbow_angle = int(angles["elbow_angle"])
    key = "FRONT" if isFront else "SIDE"
    print(elbow_angle, key)
    if (
        thresholds["SHOULDER_ELBOW"][key]["START"][0]
        <= elbow_angle
        <= thresholds["SHOULDER_ELBOW"][key]["START"][1]
    ):
        elbow = 1
    elif thresholds["SHOULDER_ELBOW"][key]["COMPLETE"] <= elbow_angle:
        elbow = 2

    return f"s{elbow}" if elbow else None


def perform_action_for_state_curl(current_state, state_tracker, angles, thresholds):
    print(current_state, state_tracker["state_seq"])
    if current_state == "s1" and len(state_tracker["state_seq"]) == 2:
        if (
            len(state_tracker["state_seq"]) == 2
            and not state_tracker["INCORRECT_POSTURE"]
        ):
            state_tracker["REP_COUNT"] += 1
        elif state_tracker["INCORRECT_POSTURE"]:
            state_tracker["IMPROPER_REP_COUNT"] += 1
            state_tracker["INCORRECT_POSTURE"] = False
        state_tracker["state_seq"] = []
        state_tracker["INCORRECT_POSTURE"] = False
    elif abs(angles["back_dist"]) > thresholds["BACK_DIST_THRESH"]:
        state_tracker["INCORRECT_POSTURE"] = True
        state_tracker["DISPLAY_TEXT"][0] = True
        print("back not straight")


def on_front_view_curl(
    coords, angles, state_tracker, frame, frame_width, frame_height, COLORS
):
    # draw_text(
    #     frame,
    #     "Camera not alligned properly: ",
    #     pos=(int(frame_width * 0.68), 30),
    #     text_color=(255, 255, 230),
    #     font_scale=0.7,
    #     text_color_bg=(18, 185, 0),
    # )

    # draw_text(
    #     frame,
    #     "INCORRECT: " + str(state_tracker["IMPROPER_REP_COUNT"]),
    #     pos=(int(frame_width * 0.68), 80),
    #     text_color=(255, 255, 230),
    #     font_scale=0.7,
    #     text_color_bg=(221, 0, 0),
    # )
    return False


def calc_curl_coords(lm, frame_width, frame_height):
    coords = get_complete_coords(lm, frame_width, frame_height)
    if (
        coords["shldr_coord"] is None
        or coords["elbow_coord"] is None
        or coords["wrist_coord"] is None
        or coords["hip_coord"] is None
    ):
        return False

    return coords


def calc_curl_angles(coords):
    offset_angle = find_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    elbow_angle = find_angle(
        coords["shldr_coord"], coords["elbow_coord"], coords["wrist_coord"]
    )
    elbow_angle_3d = find_angle_3d(
        coords["shldr_coord"], coords["elbow_coord"], coords["wrist_coord"]
    )
    hip_vertical_angle = find_angle(
        coords["shldr_coord"],
        np.array([coords["hip_coord"][0], 0]),
        coords["hip_coord"],
    )

    back_dist = coords["hip_coord"][0] - coords["shldr_coord"][0]
    return {
        "elbow_angle": elbow_angle,
        "hip_vertical_angle": hip_vertical_angle,
        "offset_angle": offset_angle,
        "elbow_angle_3d": elbow_angle_3d,
        "back_dist": back_dist,
    }


def update_state_sequence_curl(state, state_tracker):
    print(state, state_tracker["state_seq"])
    if state == "s1":
        if len(state_tracker["state_seq"]) == 0:
            state_tracker["state_seq"].append(state)
    elif state == "s2":
        if len(state_tracker["state_seq"]) == 1 and "s1" in state_tracker["state_seq"]:
            print("appending s2")
            state_tracker["state_seq"].append(state)


def on_side_view_curl(coords, angles, COLORS, frame, linetype, font):
    # Safely extract coordinates, handling None values
    def safe_tuple(coord):
        return tuple(map(int, coord[:2])) if coord is not None else None

    shldr_coord = safe_tuple(coords.get("shldr_coord"))
    elbow_coord = safe_tuple(coords.get("elbow_coord"))
    wrist_coord = safe_tuple(coords.get("wrist_coord"))
    hip_coord = safe_tuple(coords.get("hip_coord"))

    multiplier = coords.get("multiplier", 1)

    # ------------------- Bicep Curl Angles -------------------
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
    draw_angle_marker(hip_coord, angles.get("hip_vertical_angle"), COLORS["white"])

    # ------------------- Connect Body Joints -------------------
    def draw_line(coord1, coord2):
        if coord1 and coord2:
            cv2.line(frame, coord1, coord2, COLORS["light_blue"], 4, lineType=linetype)

    draw_line(shldr_coord, elbow_coord)
    draw_line(elbow_coord, wrist_coord)
    draw_line(shldr_coord, hip_coord)  # Checking posture

    # ------------------- Plot Key Landmarks -------------------
    def draw_point(coord):
        if coord:
            cv2.circle(frame, coord, 7, COLORS["yellow"], -1, lineType=linetype)

    draw_point(shldr_coord)
    draw_point(elbow_coord)
    draw_point(wrist_coord)
    draw_point(hip_coord)

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
    draw_text(hip_coord, angles.get("hip_vertical_angle"))

    return True


curl_config = ExerciseConfig(
    coords_calc_fn=calc_curl_coords,
    angles_calc_fn=calc_curl_angles,
    thresholds_fn=get_thresholds_curl,
    get_state_fn=get_curl_state,
    perform_action_fn=perform_action_for_state_curl,
    view_fns={"front": on_front_view_curl, "side": on_side_view_curl},
    feedback_map={
        0: ("STRAIGHTEN BACK", 215, (0, 153, 255)),
        1: ("BEND FORWARD", 215, (0, 153, 255)),
    },
    update_state_sequence=update_state_sequence_curl,
    name="bicep-curl",
)
