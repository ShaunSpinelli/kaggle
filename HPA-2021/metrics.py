# --- 100 characters ------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import confusion_matrix


class Metric:
    """Base class for metrics"""
    def __init__(self):
        self.running_total = 0
        self.call_count = 0

    def __call__(self, predictions, labels, ):
        """Calculate streaming result"""
        self.call_count += 1
        res = self.calculation(predictions, labels)
        self.running_total += res
        return self.running_total/self.call_count

    def calculation(self, predictions, labels):
        """Calculation implementation"""
        raise NotImplementedError

    def reset(self):
        """Reset Streaming Metrics"""
        self.running_total = 0
        self.call_count = 0


class MeanAP(Metric):
    """Mean Average Precision"""

    def calculation(self, predictions, labels):
        preds_np = np.argmax(predictions.numpy(), axis=1).flatten()
        labels_np = labels.numpy().flatten()
        tn, fp, fn, tp = confusion_matrix(labels_np, preds_np)
        return tp/ (tp+fp)

