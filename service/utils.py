from typing import List

import albumentations as A
import cv2
import numpy as np
import streamlit as st
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
    tolerance: float = 0.1,
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

    # print(embedding_to_compare)

    # print(whale_embeddings)

    embedding_to_compare = np.array(embedding_to_compare)

    whales_distances = np.linalg.norm(embedding_to_compare - whale_embeddings, axis=1)

    best_match_index = np.argmin(whales_distances)

    whales_distances = np.min(whales_distances)

    print(whales_distances)

    print(best_match_index)

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


def extractor_from_input_image(image: np.array, model, device):

    transform = A.Compose(
        [
            Letterbox((224, 224)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    transformed_image = transform(image=image)["image"]
    # cv2.imshow("123", transformed_image)
    # cv2.waitKey(0)
    images = transformed_image.unsqueeze_(0).cuda()
    with torch.no_grad():
        model.eval()
        outputs = model(images)
        outputs = outputs.cpu().numpy()

    print(outputs)

    return outputs
