import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from whale_recognition.utils import Letterbox

train_transform = A.Compose(
    [
        Letterbox((224, 224)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def compute_mean_embedding(model, images, device, old_embedding=None):
    images = [train_transform(image=image)["image"] for image in images]
    images = torch.stack(images)

    with torch.no_grad():
        outputs = model(images.to(device)).to(device)
    embeddings = outputs.cpu().detach().numpy()
    # mean_embedding = sum(embeddings) / len(embeddings)
    mean_embedding = embeddings
    if old_embedding is None:
        return mean_embedding
    else:
        return sum([mean_embedding, old_embedding]) / 2

    # return mean_embedding.to_list()
    # return np.expand_dims(mean_embedding, axis=0)


# def get_embedding_from_image(image: np.array, model, device):

#     transoformed = train_transform(image=image)
#     transoformed_image = transoformed["image"]
#     outputs = model(torch.unsqueeze(transoformed_image, 0).to(device))
#     outputs = outputs.cpu()
#     embedding = outputs.detach().numpy()

#     return embedding
