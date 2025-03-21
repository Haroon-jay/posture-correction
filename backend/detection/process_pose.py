import cv2
import numpy as np
from utils import draw_text, colors, normalize_sequences
from state_tracker import ExerciseStateTracker


class ProcessFrame:
    def __init__(self, config, flip_frame=False):

        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame
        self.config = config

        self.current_pred = None
        # self.thresholds
        print("CONFIG: ", config.thresholds)

        self.thresholds = self.config.thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = colors

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = ExerciseStateTracker(config)

        self.FEEDBACK_ID_MAP = self.config.feedback_map

    def show_rep_count(
        self,
        frame,
        frame_width,
    ):
        draw_text(
            frame,
            "CORRECT: " + str(self.state_tracker["REP_COUNT"]),
            pos=(int(frame_width * 0.68), 30),
            text_color=(255, 255, 230),
            font_scale=0.7,
            text_color_bg=(18, 185, 0),
        )

        draw_text(
            frame,
            "INCORRECT: " + str(self.state_tracker["IMPROPER_REP_COUNT"]),
            pos=(int(frame_width * 0.68), 80),
            text_color=(255, 255, 230),
            font_scale=0.7,
            text_color_bg=(221, 0, 0),
        )

    def _show_feedback(self, frame, c_frame, dict_maps):
        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2],
            )

        return frame

    def process(self, landmarks, frame, frame_width, frame_height):
        play_sound = None

        if landmarks is not None:
            # return frame, play_sound
            coords = self.config.coords_calc_fn(landmarks, frame_width, frame_height)

            if not coords:
                play_sound = "landmarks_not_visible"
                # print("Landmarks not visible enough")
                # draw_text(
                #     frame,
                #     "BODY NOT COMPLETELY VISIBLE!!!",
                #     pos=(30, frame_height - 150),
                #     text_color=(255, 255, 230),
                #     font_scale=0.65,
                #     text_color_bg=(255, 153, 0),
                # )

                # self.show_rep_count(frame)
                return self.state_tracker, play_sound

            # elbow_angle = calculate_angle_new(
            #     coords["shldr_coord"],
            #     coords["elbow_coord"],
            #     coords["wrist_coord"],
            # )

            # draw_text(
            #     frame,
            #     "Elbow Angle: " + str(elbow_angle),
            #     pos=(30, frame_height - 150),
            #     text_color=(255, 255, 230),
            #     font_scale=0.65,
            #     text_color_bg=(255, 153, 0),
            # )
            # return frame, play_sound
            angles = self.config.angles_calc_fn(coords)

            offset_angle = angles["offset_angle"]
            if offset_angle is None:
                play_sound = "landmarks_not_visible"
                return None, play_sound

            if offset_angle > self.thresholds["OFFSET_THRESH"]:
                print("front view")
                accepted = self.config.on_front_view_fn(
                    coords,
                    angles,
                    self.state_tracker,
                    frame,
                    frame_width,
                    frame_height,
                    self.COLORS,
                )
                if accepted:
                    current_state = self.config.get_state_fn(
                        angles,
                        self.thresholds,
                    )
                    self.state_tracker["curr_state"] = current_state
                    self.config.update_state_sequence(current_state, self.state_tracker)

                    self.config.perform_action_fn(
                        current_state, self.state_tracker, angles, self.thresholds
                    )

                    self.state_tracker["COUNT_FRAMES"][
                        self.state_tracker["DISPLAY_TEXT"]
                    ] += 1

                    # frame = self._show_feedback(
                    #     frame, self.state_tracker["COUNT_FRAMES"], self.FEEDBACK_ID_MAP
                    # )

                    self.state_tracker["DISPLAY_TEXT"][
                        self.state_tracker["COUNT_FRAMES"]
                        > self.thresholds["CNT_FRAME_THRESH"]
                    ] = False
                    self.state_tracker["COUNT_FRAMES"][
                        self.state_tracker["COUNT_FRAMES"]
                        > self.thresholds["CNT_FRAME_THRESH"]
                    ] = 0
                    self.state_tracker["prev_state"] = current_state
                else:
                    return self.state_tracker, "side_view_required"

            # Side view
            else:

                accepted = self.config.on_side_view_fn(
                    coords, angles, self.COLORS, frame, self.linetype, self.font
                )
                if accepted:

                    current_state = self.config.get_state_fn(
                        angles, self.thresholds, isFront=False
                    )
                    self.state_tracker["curr_state"] = current_state
                    self.config.update_state_sequence(current_state, self.state_tracker)
                    # self._update_state_sequence(current_state)

                    self.config.perform_action_fn(
                        current_state, self.state_tracker, angles, self.thresholds
                    )

                    self.state_tracker["COUNT_FRAMES"][
                        self.state_tracker["DISPLAY_TEXT"]
                    ] += 1

                    # frame = self._show_feedback(
                    #     frame, self.state_tracker["COUNT_FRAMES"], self.FEEDBACK_ID_MAP
                    # )

                    self.state_tracker["DISPLAY_TEXT"][
                        self.state_tracker["COUNT_FRAMES"]
                        > self.thresholds["CNT_FRAME_THRESH"]
                    ] = False
                    self.state_tracker["COUNT_FRAMES"][
                        self.state_tracker["COUNT_FRAMES"]
                        > self.thresholds["CNT_FRAME_THRESH"]
                    ] = 0
                    self.state_tracker["prev_state"] = current_state

        else:

            # Reset all other state variables
            self.state_tracker["prev_state"] = None
            self.state_tracker["curr_state"] = None
            # self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker["INCORRECT_POSTURE"] = False
            self.state_tracker["DISPLAY_TEXT"] = np.full((5,), False)
            self.state_tracker["COUNT_FRAMES"] = np.zeros((5,), dtype=np.int64)
            # self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        # self.show_rep_count(frame, frame_width)

        # draw_text(
        #     frame,
        #     "C-STATE: " + str(self.state_tracker["curr_state"]),
        #     pos=(30, frame_height - 400),
        #     text_color=(255, 255, 230),
        # )
        # draw_text(
        #     frame,
        #     "STATE SEQ: " + str(self.state_tracker["state_seq"]),
        #     pos=(30, frame_height - 350),
        #     text_color=(255, 255, 230),
        # )
        feedback_msg = play_sound
        for idx, display in enumerate(self.state_tracker["DISPLAY_TEXT"]):
            if display:
                feedback_msg = self.FEEDBACK_ID_MAP[idx][0]
                break

        return self.state_tracker, feedback_msg
