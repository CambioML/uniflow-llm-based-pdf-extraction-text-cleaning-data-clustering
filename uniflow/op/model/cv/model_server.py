"""All CV Model Servers including NougatModelServer, LayoutModelServer."""

import logging
import re
from typing import Any, Dict, List

from uniflow.op.model.model_config import LayoutModelConfig, NougatModelConfig
from uniflow.op.model.model_server import AbsModelServer
from uniflow.op.prompt import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
###############################################################################
#                             All CV Model Servers                            #
###############################################################################


class NougatModelServer(AbsModelServer):
    """Nougat Model Server Class."""

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        # import in class level to avoid installing nougat package
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install nougat to use NougatModelServer. You can use `pip install transformers` to install it."
            ) from exc

        super().__init__(prompt_template, model_config)
        self._model_config = NougatModelConfig(**self._model_config)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = NougatProcessor.from_pretrained(
            self._model_config.model_name, torch_dtype=self.dtype
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self._model_config.model_name, torch_dtype=self.dtype
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.eval()

    def _preprocess(self, data: str) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [d["generated_text"] for output_list in data for d in output_list]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """

        import pypdfium2  # pylint: disable=import-outside-toplevel

        outs = []
        for pdf in data:
            pdf = pypdfium2.PdfDocument(pdf)
            pages = range(len(pdf))
            renderer = pdf.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=pages,
                scale=96 / 72,
            )
            images = []
            for i, image in zip(pages, renderer):
                images.append(image)
            predictions = []
            for start_idx in range(0, len(images), self._model_config.batch_size):
                batch = images[start_idx : start_idx + self._model_config.batch_size]
                pixel_values = (
                    self.processor(batch, return_tensors="pt")
                    .to(self.dtype)
                    .pixel_values
                )
                outputs = self.model.generate(
                    pixel_values.to(self.device),
                    min_length=1,
                    max_new_tokens=3584,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,  # pylint: disable=no-member
                    eos_token_id=self.processor.tokenizer.eos_token_id,  # pylint: disable=no-member
                    do_sample=False,
                    bad_words_ids=[
                        [
                            self.processor.tokenizer.unk_token_id  # pylint: disable=no-member
                        ]
                    ],
                )
                sequence = self.processor.batch_decode(
                    outputs, skip_special_tokens=True
                )  # [0]
                sequence = self.processor.post_process_generation(
                    sequence, fix_markdown=False
                )
                predictions.extend(sequence)
            out = "\n\n".join(predictions).strip()
            out = re.sub(r"\n{3,}", "\n\n", out).strip()
            outs.append(out)
        return outs


class LayoutModelServer(AbsModelServer):
    """Layout Model Server Class."""

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        super().__init__(prompt_template, model_config)
        self._model_config = LayoutModelConfig(**self._model_config)
        try:
            import easyocr  # pylint: disable=import-outside-toplevel

            self.reader = easyocr.Reader(self._model_config.ocr_lang)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install easyocr to use LayoutModelServer. You can use `pip install easyocr` to install it."
            ) from exc
        from .layout_utils import (  # pylint: disable=import-outside-toplevel
            LayoutPredictor,
        )

        self.layout_predictor = LayoutPredictor(
            self._model_config.model_name, self._model_config.model_file
        )

    def _preprocess(self, data: str) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [d["generated_text"] for output_list in data for d in output_list]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        from uniflow.op.model.cv.layout_utils import (  # pylint: disable=import-outside-toplevel
            XYCut,
        )

        outs = []
        for img in data:
            img = cv2.imread(img)
            ori_im = img.copy()
            h, w, _ = img.shape
            layout_res = self.layout_predictor(img)
            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)

                wht_im[y1:y2, x1:x2, :] = roi_img
                result = self.reader.readtext(wht_im)
                if len(result) == 0:
                    continue
                filter_boxes, filter_rec_res, scores = zip(*result)
                res = []
                for box, rec_res, score in zip(filter_boxes, filter_rec_res, scores):
                    rec_str = rec_res
                    rec_conf = score
                    res.append(
                        {
                            "text": rec_str,
                            "confidence": float(rec_conf),
                            "text_region": box,
                        }
                    )
                res_list.append(
                    {
                        "type": region["type"].lower(),
                        "bbox": [x1, y1, x2, y2],
                        "img": roi_img,
                        "res": res,
                    }
                )
            res = []
            boxes = [res["bbox"] for res in res_list]
            XYCut.recursive_xy_cut(
                np.asarray(boxes).astype(int), np.arange(len(boxes)), res
            )
            sorted_res_list = [res_list[idx] for idx in res]
            final_md = ""
            for _, region in enumerate(sorted_res_list):
                if len(region["res"]) == 0:
                    continue
                if region["type"] in ("title", "page-header", "section-header"):
                    final_md += (
                        "## "
                        + " ".join([text["text"] for text in region["res"]])
                        + "\n\n"
                    )
                elif region["type"] in (
                    "picture",
                    "footnote",
                    "formula",
                    "list-item",
                    "text",
                    "caption",
                    "page-footer",
                    "table",
                ):
                    final_md += (
                        " ".join([text["text"] for text in region["res"]]) + "\n\n"
                    )
                else:
                    print(region["type"])
            out = re.sub(r"\n{3,}", "\n\n", final_md.strip()).strip()
            outs.append(out)
        return outs
