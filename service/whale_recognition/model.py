import torch


def get_model():
    model = torch.jit.load("./whale_recognition/model_0004_pairs_1.0000.pth")
    model.eval()
    return model
