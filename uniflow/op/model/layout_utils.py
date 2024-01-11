"""Layout Module."""
from typing import List

import cv2
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download


class Layout:
    """
    Layout class.
    """

    @staticmethod
    def preprocess(
        img: np.ndarray, input_size: tuple, swap: tuple = (2, 0, 1)
    ) -> tuple[np.ndarray, float]:
        """
        Preprocesses the input image by resizing and padding it.

        Args:
            img (numpy.ndarray): The input image.
            input_size (tuple): The desired size of the input image.
            swap (tuple, optional): The order of dimensions to swap. Defaults to (2, 0, 1).

        Returns:
            tuple[numpy.ndarray, float]: The preprocessed image and the scaling factor applied to the image.
        """
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> List[int]:
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def multiclass_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        nms_thr: float,
        score_thr: float,
        class_agnostic: bool = True,
    ) -> np.ndarray:
        """Multiclass NMS implemented in Numpy"""
        if class_agnostic:
            nms_method = Layout.multiclass_nms_class_agnostic
        else:
            nms_method = Layout.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)

    @staticmethod
    def multiclass_nms_class_aware(
        boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> np.ndarray:
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = Layout.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    @staticmethod
    def multiclass_nms_class_agnostic(
        boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> np.ndarray:
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = Layout.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )
        return dets

    @staticmethod
    def postprocess(
        outputs: np.ndarray, img_size: tuple, p6: bool = False
    ) -> np.ndarray:
        """
        Postprocesses the outputs of a model by applying grid and stride calculations.

        Args:
            outputs (numpy.ndarray): The output predictions from the model.
            img_size (tuple): The size of the input image.
            p6 (bool, optional): Whether to include the stride 64 grid. Defaults to False.

        Returns:
            numpy.ndarray: The postprocessed outputs.

        """
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs


class XYCut:
    """
    XYCut class.
    """

    @staticmethod
    def projection_by_bboxes(boxes: np.array, axis: int) -> np.ndarray:
        """
        Get the projection histogram by a set of bounding boxes, and output it in per-pixel form.

        Args:
            boxes: [N, 4]
            axis: 0 for horizontal projection along the x-axis, 1 for vertical projection along the y-axis

        Returns:
            1D projection histogram with a length equal to the maximum value of the coordinate in the projection direction
            (We don't need the actual image size, as we are only interested in finding the gaps between text boxes)
        """
        assert axis in [0, 1]
        length = np.max(boxes[:, axis::2])
        res = np.zeros(length, dtype=int)
        for start, end in boxes[:, axis::2]:
            res[start:end] += 1
        return res

    @staticmethod
    def split_projection_profile(
        arr_values: np.array, min_value: float, min_gap: float
    ):
        """Split projection profile:
        from: https://dothinking.github.io/2021-06-19-%E9%80%92%E5%BD%92%E6%8A%95%E5%BD%B1%E5%88%86%E5%89%B2%E7%AE%97%E6%B3%95/#:~:text=%E9%80%92%E5%BD%92%E6%8A%95%E5%BD%B1%E5%88%86%E5%89%B2%EF%BC%88Recursive%20XY,%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%88%92%E5%88%86%E6%AE%B5%E8%90%BD%E3%80%81%E8%A1%8C%E3%80%82


        ```
                                ┌──┐
            arr_values           │  │       ┌─┐───
                ┌──┐             │  │       │ │ |
                │  │             │  │ ┌───┐ │ │min_value
                │  │<- min_gap ->│  │ │   │ │ │ |
            ────┴──┴─────────────┴──┴─┴───┴─┴─┴─┴───
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        ```

        Args:
            arr_values (np.array): 1-d array representing the projection profile.
            min_value (float): Ignore the profile if `arr_value` is less than `min_value`.
            min_gap (float): Ignore the gap if less than this value.

        Returns:
            tuple: Start indexes and end indexes of split groups.
        """
        # all indexes with projection height exceeding the threshold
        arr_index = np.where(arr_values > min_value)[0]
        if len(arr_index) == 0:
            return

        # find zero intervals between adjacent projections
        # |  |                    ||
        # ||||<- zero-interval -> |||||
        arr_diff = arr_index[1:] - arr_index[0:-1]
        arr_diff_index = np.where(arr_diff > min_gap)[0]
        arr_zero_intvl_start = arr_index[arr_diff_index]
        arr_zero_intvl_end = arr_index[arr_diff_index + 1]

        # convert to index of projection range:
        # the start index of zero interval is the end index of projection
        arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
        arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
        arr_end += 1  # end index will be excluded as index slice

        return arr_start, arr_end

    @staticmethod
    def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
        """
        from https://github.com/Sanster/xy-cut/blob/main/xycut.py
        Args:
            boxes: (N, 4)
            indices: Represents the indices of the boxes in the original data during the recursive process.
            res: Stores the output results.

        """
        assert len(boxes) == len(indices)

        _indices = boxes[:, 1].argsort()
        y_sorted_boxes = boxes[_indices]
        y_sorted_indices = indices[_indices]

        # debug_vis(y_sorted_boxes, y_sorted_indices)

        y_projection = XYCut.projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
        pos_y = XYCut.split_projection_profile(y_projection, 0, 1)
        if not pos_y:
            return

        arr_y0, arr_y1 = pos_y
        for r0, r1 in zip(arr_y0, arr_y1):
            _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

            y_sorted_boxes_chunk = y_sorted_boxes[_indices]
            y_sorted_indices_chunk = y_sorted_indices[_indices]

            _indices = y_sorted_boxes_chunk[:, 0].argsort()
            x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
            x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]

            x_projection = XYCut.projection_by_bboxes(
                boxes=x_sorted_boxes_chunk, axis=0
            )
            pos_x = XYCut.split_projection_profile(x_projection, 0, 1)
            if not pos_x:
                continue

            arr_x0, arr_x1 = pos_x
            if len(arr_x0) == 1:
                res.extend(x_sorted_indices_chunk)
                continue

            for c0, c1 in zip(arr_x0, arr_x1):
                _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (
                    x_sorted_boxes_chunk[:, 0] < c1
                )
                XYCut.recursive_xy_cut(
                    x_sorted_boxes_chunk[_indices],
                    x_sorted_indices_chunk[_indices],
                    res,
                )


class LayoutPredictor:
    """
    Class for predicting layout of an image.

    Attributes:
        ort_session (onnxruntime.InferenceSession): Inference session for running the model.
        categorys (dict): Mapping of category index to category name.
    """

    H = 1024
    W = 768

    def __init__(self, model_name: str, model_file: str):
        """
        Initializes the LayoutPredictor class.
        Downloads the model specified by `model_name` and `model_file` from the Hugging Face Model Hub.
        Creates an inference session using the downloaded model.
        """
        model_path = hf_hub_download(model_name, model_file)
        self.ort_session = onnxruntime.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.categorys = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
        }

    def __call__(self, img):
        """
        Predicts the layout of the input image.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            list: List of dictionaries containing the predicted layout information.
                Each dictionary contains the type of the element and its bounding box coordinates.
        """
        image, ratio = Layout.preprocess(img, (self.H, self.W))
        res = self.ort_session.run(["output"], {"images": image[np.newaxis, :]})[0]
        predictions = Layout.postprocess(res, (self.H, self.W), p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = Layout.multiclass_nms(boxes_xyxy, scores, nms_thr=0.15, score_thr=0.3)
        result = []
        if dets is None:
            return result
        scores = dets[:, 4]
        final_cls_inds = dets[:, 5]
        final_boxes = dets[:, :4].astype("int")

        for box_idx, box in enumerate(final_boxes):
            result.append(
                {"type": self.categorys[int(final_cls_inds[box_idx])], "bbox": box}
            )
        return result
