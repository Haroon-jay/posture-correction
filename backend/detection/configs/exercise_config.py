class ExerciseConfig:
    """Configuration Class for different exercises"""

    def __init__(
        self,
        coords_calc_fn,
        angles_calc_fn,
        thresholds_fn,
        get_state_fn,
        perform_action_fn,
        view_fns,
        feedback_map,
        update_state_sequence,
        name,
    ):
        self.coords_calc_fn = coords_calc_fn
        self.angles_calc_fn = angles_calc_fn
        print("thresholds_fn", thresholds_fn())
        self.thresholds = thresholds_fn()
        self.get_state_fn = get_state_fn
        self.perform_action_fn = perform_action_fn
        self.on_front_view_fn = view_fns["front"]
        self.on_side_view_fn = view_fns["side"]
        self.feedback_map = feedback_map
        self.update_state_sequence = update_state_sequence
        self.name = name
