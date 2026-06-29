import numpy as np
import torch
from tqdm import tqdm


class XRayEvaluator:
    """
    Evaluation engine for multi-label Chest X-ray classification.

    Responsibilities:
    - run inference
    - collect y_true, y_prob, y_pred
    - support checkpoint-loaded models
    """

    def __init__(
        self,
        model,
        device,
        threshold: float = 0.5,
        logger=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.logger = logger

    def _log(self, message: str):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    @torch.no_grad()
    def predict(self, dataloader):
        """
        Run inference on a dataloader.

        Returns
        -------
        dict
            {
                "y_true": np.ndarray,
                "y_prob": np.ndarray,
                "y_pred": np.ndarray
            }
        """

        self.model.eval()

        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.float()

            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu()

            all_labels.append(labels.cpu())
            all_probs.append(probs)

        y_true = torch.cat(all_labels, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()
        y_pred = (y_prob >= self.threshold).astype(np.int32)

        return {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

    def evaluate(
        self,
        dataloader,
        metric_fn=None,
    ):
        """
        Predict and optionally compute metrics.
        """

        outputs = self.predict(dataloader)

        if metric_fn is not None:
            metrics = metric_fn(
                outputs["y_true"],
                outputs["y_prob"],
            )
            outputs["metrics"] = metrics

        return outputs