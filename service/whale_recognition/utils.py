import json
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class Letterbox(object):
    def __init__(
        self,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        *kwargs
    ):
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup

    def __call__(self, image, *kwargs):
        shape = image.shape  # current shape [width, height]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            self.new_shape[1] - new_unpad[0],
            self.new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])
            ratio = (
                self.new_shape[1] / shape[1],
                self.new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color
        )  # add border
        return {"image": image}


def bytes_to_numpy(image: bytes):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image


def calculate_embed(
    whale_ids: list,
    whale_embeddings: List[List[float]],
    embedding_to_compare: List[float],
    tolerance: float = 0.5,
):
    """
    Сравнение признаков китов со списком признаков китов известных китов из БД.

    :param face_ids: Список идентификаторов китов для сравнения.
    :param face_embeddings: Список признаков китов для сравнения.
    :param embedding_to_compare: Признаки сравниваемого китов.
    :param tolerance: Порог "расстояния" между китами, при котором лицо признаётся распознанным.
    :return: Возвращает идентификатор + score кита или None в случае неудачи.
    """
    # print(embedding_to_compare-whale_embeddings)
    result = {"status": False}

    embedding_to_compare = np.asarray(embedding_to_compare)

    whales_distances = np.linalg.norm(embedding_to_compare - whale_embeddings, axis=1)

    # print(whales_distances)

    best_match_index = np.argmin(whales_distances)

    # print("========= embeding to compare")
    # print(len(embedding_to_compare))
    # print("========= best simularity embedding")
    # print(print(whale_embeddings[best_match_index]))
    # print(f"LEN DB {len(whale_embeddings)}")
    # print("INDEX", best_match_index)

    whales_distances = np.min(whales_distances)

    if whales_distances < tolerance:
        result["status"] = True
        result["whale_id"] = whale_ids[best_match_index]
        result["whales_distances"] = whales_distances
        # confidence = face_distance_to_conf(whales_distances)
        # result["confidence"] = confidence
        return result
    else:
        result["whale_id"] = whale_ids[best_match_index]
        result["whales_distances"] = whales_distances
        # confidence = face_distance_to_conf(whales_distances)
        # result["confidence"] = confidence
        return result


def get_metadata(path: Path):
    if path.exists():
        with path.open(mode="r") as f:
            return json.load(f)
    return {}


def get_numpy_db(path: Path):
    if path.exists():
        return np.load(path.__str__())
    return np.empty(shape=(1, 512))
