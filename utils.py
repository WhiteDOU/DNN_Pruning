import torch
import pickle
MODEL_PATH = 'model_dict.pkl'
def load_model_dict(model_path):
    with open(model_path, 'rb') as f:
        model_dict = torch.load(f)
    return model_dict


def save_model_dict(model):
    model_dict = model.state_dict()
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)