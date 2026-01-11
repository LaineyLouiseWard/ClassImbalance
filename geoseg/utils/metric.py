import numpy as np


class Evaluator(object):
    def __init__(self, num_class: int, ignore_index=None):
        """
        Args:
            num_class: number of classes (0..num_class-1)
            ignore_index: label value to ignore in metrics (e.g. 255 for void).
                          Can be outside [0, num_class-1].
        """
        self.num_class = int(num_class)
        self.ignore_index = ignore_index
        self.eps = 1e-8
        self.reset()

    def _valid_mask(self, gt_image: np.ndarray, pre_image: np.ndarray | None = None) -> np.ndarray:
        """
        Valid pixels are those whose GT label is in [0, num_class-1],
        excluding ignore_index (even if ignore_index is outside range).

        Optionally also enforce predictions in range (defensive).
        """
        # GT must be in-range
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        # Explicitly drop ignore_index if provided (covers ignore_index=255 too)
        if self.ignore_index is not None:
            mask = mask & (gt_image != self.ignore_index)

        # Optional: ensure predictions are also in-range
        if pre_image is not None:
            mask = mask & (pre_image >= 0) & (pre_image < self.num_class)

        return mask

    def _generate_matrix(self, gt_image: np.ndarray, pre_image: np.ndarray) -> np.ndarray:
        mask = self._valid_mask(gt_image, pre_image)
        gt = gt_image[mask].astype(np.int64)
        pr = pre_image[mask].astype(np.int64)

        # Fast bincount confusion
        label = self.num_class * gt + pr
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image: np.ndarray, pre_image: np.ndarray):
        assert gt_image.shape == pre_image.shape, (
            f"pre_image shape {pre_image.shape}, gt_image shape {gt_image.shape}"
        )
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.float64)

    # ---- metrics ----
    def get_tp_fp_tn_fn(self):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + self.eps)

    def Recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fn + self.eps)

    def F1(self):
        p = self.Precision()
        r = self.Recall()
        return (2.0 * p * r) / (p + r + self.eps)

    def OA(self):
        # Overall accuracy computed on INCLUDED pixels only (since CM built on included pixels)
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)

    def Intersection_over_Union(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + fn + self.eps)

    def Dice(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return (2 * tp) / (2 * tp + fp + fn + self.eps)

    def Frequency_Weighted_Intersection_over_Union(self):
        cm = self.confusion_matrix
        freq = cm.sum(axis=1) / (cm.sum() + self.eps)
        iou = self.Intersection_over_Union()
        return (freq[freq > 0] * iou[freq > 0]).sum()

    # ---- helpers for reporting ----
    def mean_iou(self, exclude_indices=None):
        """Convenience: mean IoU excluding certain class ids (e.g. exclude_indices=[0])."""
        iou = self.Intersection_over_Union()
        if exclude_indices is None:
            return np.nanmean(iou)
        mask = np.ones_like(iou, dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < len(iou):
                mask[idx] = False
        return np.nanmean(iou[mask])

    def mean_f1(self, exclude_indices=None):
        f1 = self.F1()
        if exclude_indices is None:
            return np.nanmean(f1)
        mask = np.ones_like(f1, dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < len(f1):
                mask[idx] = False
        return np.nanmean(f1[mask])
