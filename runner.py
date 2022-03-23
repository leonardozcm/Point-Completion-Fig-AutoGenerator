import torch


class candiModel(object):
    def __init__(self, name, path, model) -> None:
        self.name = name
        self.path = path
        self.model = model

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model'])
    
    def __call__(self, data, gt):
        return self.model(data)
