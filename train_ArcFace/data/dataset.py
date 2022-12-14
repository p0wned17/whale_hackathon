import os
from pathlib import Path

import cv2
import pandas as pd
import torch.utils.data as data
from PIL import Image


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class WhalesDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        transforms,
        is_cropped: bool = False,
        is_inference=False,
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file).values.tolist()
        self.transforms = transforms
        self.is_inference = is_inference
        self.is_cropped = is_cropped

    def __getitem__(self, index):
        cv2.setNumThreads(12)

        if self.is_inference:
            if self.is_cropped:
                impath, target = self.imlist[index]
                full_imname = f"{self.root}{impath}"
            else:
                impath, x1, y1, x2, y2 = self.imlist[index]
                full_imname = os.path.join(self.root, impath)
        else:
            if self.is_cropped:
                impath, target = self.imlist[index]
                full_imname = os.path.join(self.root, impath)
            else:
                impath, x1, y1, x2, y2, target = self.imlist[index]
                full_imname = os.path.join(self.root, impath)

        if not os.path.exists(full_imname):
            print("No file ", full_imname)

        img = read_image(full_imname)
        if not self.is_cropped:
            x1, y1 = int(round(x1)), int(round(y1))
            x2, y2 = int(round(x2)), int(round(y2))

            if (
                0 <= x1 < x2
                and 0 <= y1 < y2
                and 0 <= x2 < img.shape[1]
                and 0 <= y2 < img.shape[0]
            ):
                img = img[y1:y2, x1:x2]
        img = self.transforms(image=img)["image"]

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


class ValDataset(data.Dataset):
    def __init__(self, val_path, val_pairs, val_list, transforms, is_cropped):
        self.val_path = Path(val_path)
        self.val_pairs = pd.read_csv(val_pairs)
        val_list = pd.read_csv(val_list)
        self.transforms = transforms
        self.is_cropped = is_cropped

    def __getitem__(self, index):
        cv2.setNumThreads(12)

        impath1, impath2, score = self.val_pairs.loc[index]

        if not self.is_cropped:
            _, x_1, y_1, x_2, y_2, _ = self.val_bbox.loc[
                self.val_bbox["img_path"] == impath1
            ].iloc[0]
            _, z_1, w_1, z_2, w_2, _ = self.val_bbox.loc[
                self.val_bbox["img_path"] == impath2
            ].iloc[0]

        full_imname1 = self.val_path / impath1
        full_imname2 = self.val_path / impath2

        if (not os.path.exists(full_imname1)) or (not os.path.exists(full_imname2)):
            print("No file", full_imname1, "or", full_imname2)

        img1 = read_image(str(full_imname1))
        img2 = read_image(str(full_imname2))

        if not self.is_cropped:
            x_1, y_1, x_2, y_2 = (
                int(round(x_1)),
                int(round(y_1)),
                int(round(x_2)),
                int(round(y_2)),
            )
            z_1, w_1, z_2, w_2 = (
                int(round(z_1)),
                int(round(w_1)),
                int(round(z_2)),
                int(round(w_2)),
            )

            if (
                0 <= x_1 < x_2
                and 0 <= y_1 < y_2
                and 0 <= x_2 < img1.shape[1]
                and 0 <= y_2 < img1.shape[0]
            ):
                img1 = img1[y_1:y_2, x_1:x_2]
            if (
                0 <= z_1 < z_2
                and 0 <= w_1 < w_2
                and 0 <= z_2 < img2.shape[1]
                and 0 <= w_2 < img2.shape[0]
            ):
                img2 = img2[w_1:w_2, z_1:z_2]

        # cv2.imshow('1', img1)
        # cv2.imshow('2', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.transforms is not None:
            img1 = self.transforms(image=img1)["image"]
            img2 = self.transforms(image=img2)["image"]
        return img1, img2, score

    def __len__(self):
        return len(self.val_pairs)
