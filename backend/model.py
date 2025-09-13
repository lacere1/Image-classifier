import torch
from torchvision.models import resnet50, ResNet50_Weights

class ModelWrapper:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        self.model.to(self.device)
        # Preprocessing pipeline associated with these weights
        self.preprocess = self.weights.transforms()

    def predict_topk(self, image, k: int = 5):
        import torch.nn.functional as F
        # image: PIL.Image in RGB
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # (1, 1000)
            probs = F.softmax(logits, dim=1)[0]  # (1000,)
            topk = torch.topk(probs, k)
        labels = [self.weights.meta["categories"][i] for i in topk.indices.tolist()]
        scores = topk.values.tolist()
        return list(zip(labels, scores))
