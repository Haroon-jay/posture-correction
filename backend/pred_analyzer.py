class PredAnalyzer:
    def __init__(self, history_size=5, consistency_threshold=0.7):
        self.history = []  # To store the last `history_size` predictions
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold

    def update_history(self, prediction):
        self.history.append(prediction)
        if len(self.history) > self.history_size:
            self.history.pop(0)  # Remove the oldest prediction if history is full

    def analyze(self, current_prediction):
        # Update history with the current prediction
        self.update_history(current_prediction)

        # Calculate the consistency of predictions
        prediction_counts = {}
        for pred in self.history:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

        # Find the most common prediction in the history
        most_common_prediction = max(prediction_counts, key=prediction_counts.get)
        consistency = prediction_counts[most_common_prediction] / len(self.history)

        # If consistency exceeds the threshold, trust the most common prediction
        if consistency >= self.consistency_threshold:
            return most_common_prediction
        else:
            # If the history is inconsistent, trust the current prediction
            return current_prediction
