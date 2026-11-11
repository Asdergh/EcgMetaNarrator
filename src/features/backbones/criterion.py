import torch
import torch.nn as nn
from typing import Dict


class AccumulativeHammingLoss(nn.Module):
    """Accumulative Hamming loss."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_sum_errors", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        errors = (preds != targets).sum().to(torch.float64)
        total = torch.tensor(targets.numel(), dtype=torch.float64, device=errors.device)
        if accum:
            self._sum_errors += errors.detach()
            self._sum_total += total.detach()
        return {"hamming_loss_b": (errors / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"hamming_loss": self._sum_errors / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_errors.zero_()
        self._sum_total.zero_()


class AccumulativeSubsetAccuracy(nn.Module):
    """Accumulative subset accuracy."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_sum_correct", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct = (preds == targets).all(dim=1).sum().to(torch.float64)
        total = torch.tensor(targets.size(0), dtype=torch.float64, device=correct.device)
        if accum:
            self._sum_correct += correct.detach()
            self._sum_total += total.detach()
        return {"subset_accuracy_b": (correct / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"subset_accuracy": self._sum_correct / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_correct.zero_()
        self._sum_total.zero_()


class AccumulativeJaccardAccuracy(nn.Module):
    """Accumulative Jaccard accuracy."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_sum_jaccard", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        intersection = (preds * targets).sum(dim=1)
        union = ((preds + targets) > 0).sum(dim=1)
        jaccard = torch.where(
            union > 0,
            intersection / union.clamp_min(1.0),
            torch.ones_like(intersection),
        )
        sum_jaccard = jaccard.sum().to(torch.float64)
        total = torch.tensor(targets.size(0), dtype=torch.float64, device=sum_jaccard.device)
        if accum:
            self._sum_jaccard += sum_jaccard.detach()
            self._sum_total += total.detach()
        return {"jaccard_accuracy_b": (sum_jaccard / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"jaccard_accuracy": self._sum_jaccard / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_jaccard.zero_()
        self._sum_total.zero_()


class AccumulativeDocumentPrecision(nn.Module):
    """Accumulative document-based precision."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_sum_precision", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp = (preds * targets).sum(dim=1)
        pred_pos = preds.sum(dim=1)
        precision = torch.where(
            pred_pos > 0,
            tp / pred_pos.clamp_min(1.0),
            torch.ones_like(tp),
        )
        sum_p = precision.sum().to(torch.float64)
        total = torch.tensor(targets.size(0), dtype=torch.float64, device=sum_p.device)
        if accum:
            self._sum_precision += sum_p.detach()
            self._sum_total += total.detach()
        return {"doc_precision_b": (sum_p / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"doc_precision": self._sum_precision / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_precision.zero_()
        self._sum_total.zero_()


class AccumulativeDocumentRecall(nn.Module):
    """Accumulative document-based recall."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_sum_recall", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp = (preds * targets).sum(dim=1)
        true_pos = targets.sum(dim=1)
        recall = torch.where(
            true_pos > 0,
            tp / true_pos.clamp_min(1.0),
            torch.ones_like(tp),
        )
        sum_r = recall.sum().to(torch.float64)
        total = torch.tensor(targets.size(0), dtype=torch.float64, device=sum_r.device)
        if accum:
            self._sum_recall += sum_r.detach()
            self._sum_total += total.detach()
        return {"doc_recall_b": (sum_r / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"doc_recall": self._sum_recall / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_recall.zero_()
        self._sum_total.zero_()


class AccumulativeDocumentF1(nn.Module):
    """Accumulative document-based F1."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("_sum_f1", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_sum_total", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp = (preds * targets).sum(dim=1)
        pred_pos = preds.sum(dim=1)
        true_pos = targets.sum(dim=1)
        precision = torch.where(pred_pos > 0, tp / pred_pos.clamp_min(self.eps),
                                torch.ones_like(tp))
        recall = torch.where(true_pos > 0, tp / true_pos.clamp_min(self.eps),
                            torch.ones_like(tp))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp_min(self.eps),
                        torch.zeros_like(denom))
        sum_f1 = f1.sum().to(torch.float64)
        total = torch.tensor(targets.size(0), dtype=torch.float64, device=sum_f1.device)
        if accum:
            self._sum_f1 += sum_f1.detach()
            self._sum_total += total.detach()
        return {"doc_f1_b": (sum_f1 / total.clamp_min(1.0)).to(logits.dtype)}

    def normalize(self) -> Dict[str, torch.Tensor]:
        return {"doc_f1": self._sum_f1 / self._sum_total.clamp_min(1.0)}

    def reset(self):
        self._sum_f1.zero_()
        self._sum_total.zero_()


class MultiBCECriterion(nn.Module):
    """BCE training loss + accumulative multi-label metrics."""

    def __init__(
        self,
        accuracy: bool = False,
        hamming: bool = False,
        f1: bool = False,
        accuracy_subset: bool = False,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_jaccard = accuracy
        self.use_hamming = hamming
        self.use_f1 = f1
        self.use_subset = accuracy_subset

        self.hamming_module = AccumulativeHammingLoss()
        self.subset_module = AccumulativeSubsetAccuracy()
        self.jaccard_module = AccumulativeJaccardAccuracy()
        self.doc_precision_module = AccumulativeDocumentPrecision()
        self.doc_recall_module = AccumulativeDocumentRecall()
        self.doc_f1_module = AccumulativeDocumentF1()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                accum: bool = False) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {"loss": self.bce(logits, targets)}
        if self.use_hamming:
            out.update(self.hamming_module(logits, targets, accum=accum))
        if self.use_subset:
            out.update(self.subset_module(logits, targets, accum=accum))
        if self.use_jaccard:
            out.update(self.jaccard_module(logits, targets, accum=accum))
        if self.use_f1:
            out.update(self.doc_precision_module(logits, targets, accum=accum))
            out.update(self.doc_recall_module(logits, targets, accum=accum))
            out.update(self.doc_f1_module(logits, targets, accum=accum))
        return out

    def normalize(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.use_hamming:
            out.update(self.hamming_module.normalize())
        if self.use_subset:
            out.update(self.subset_module.normalize())
        if self.use_jaccard:
            out.update(self.jaccard_module.normalize())
        if self.use_f1:
            out.update(self.doc_precision_module.normalize())
            out.update(self.doc_recall_module.normalize())
            out.update(self.doc_f1_module.normalize())
        return out

    def reset(self):
        self.hamming_module.reset()
        self.subset_module.reset()
        self.jaccard_module.reset()
        self.doc_precision_module.reset()
        self.doc_recall_module.reset()
        self.doc_f1_module.reset()