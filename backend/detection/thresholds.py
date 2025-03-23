def get_thresholds_pushup():
    ELBOW_ANGLE = {
        "SIDE": {"START": (0, 20), "TRANS": (40, 50), "COMPLETE": 60},
        "FRONT": {"START": (0, 15), "TRANS": (20, 40), "COMPLETE": 50},
    }
    thresholds = {
        "ELBOW_ANGLE": ELBOW_ANGLE,
        "OFFSET_THRESH": 50,
        "CNT_FRAME_THRESH": 55,
    }
    return thresholds


def get_thresholds_curl():
    _ANGLE_SHOULDER_ELBOW = {
        "FRONT": {"START": (0, 20), "COMPLETE": 170},
        "SIDE": {"START": (0, 20), "COMPLETE": 54},
    }
    thresholds = {
        "SHOULDER_ELBOW": _ANGLE_SHOULDER_ELBOW,
        "OFFSET_THRESH": 65,
        "CNT_FRAME_THRESH": 55,
        "HIP_VERT_THRESH": 15,
        "GROUND_UPPER_ARM_THRESH": 150,
    }
    return thresholds


# Get thresholds for beginner mode
def get_thresholds_beginner():

    _ANGLE_HIP_KNEE_VERT = {"NORMAL": (0, 32), "TRANS": (35, 65), "PASS": (70, 95)}

    thresholds = {
        "HIP_KNEE_VERT": _ANGLE_HIP_KNEE_VERT,
        "HIP_THRESH": [10, 50],
        "ANKLE_THRESH": 45,
        "KNEE_THRESH": [50, 70, 95],
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
        "CNT_FRAME_THRESH": 50,
    }

    return thresholds


# Get thresholds for beginner mode
def get_thresholds_pro():

    _ANGLE_HIP_KNEE_VERT = {"NORMAL": (0, 32), "TRANS": (35, 65), "PASS": (80, 95)}

    thresholds = {
        "HIP_KNEE_VERT": _ANGLE_HIP_KNEE_VERT,
        "HIP_THRESH": [15, 50],
        "ANKLE_THRESH": 30,
        "KNEE_THRESH": [50, 80, 95],
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
        "CNT_FRAME_THRESH": 50,
    }

    return thresholds
