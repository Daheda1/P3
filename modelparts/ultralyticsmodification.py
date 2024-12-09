import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss, make_anchors
from ultralytics.utils import IterableSimpleNamespace

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model, desired_classes, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        self.desired_classes = desired_classes
        self.num_classes = self.nc  # Number of classes
        self.device = self.device

        # Create a mask for desired classes
        self.class_mask = torch.zeros(self.num_classes, device=self.device)
        self.class_mask[self.desired_classes] = 1.0  # Set desired classes to 1
        self.class_mask = self.class_mask.view(1, 1, -1)  # Reshape for broadcasting

    def __call__(self, preds, batch):
        # Initialize loss components
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # Extract features and predictions
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        
        # Correctly call make_anchors as a standalone function
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Prepare targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted bounding boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Assign targets to predictions
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)        

        # Apply class mask to target_scores
        target_scores = target_scores * self.class_mask.to(target_scores.device)


        # Compute classification loss
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        loss[1] = loss_cls

        # Compute bounding box and distribution loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Apply loss coefficients
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace

class CustomDetectionModel(DetectionModel):
    def __init__(self, cfg=None, ch=3, nc=None, desired_classes=None, args=None, verbose=False):
        super().__init__(cfg, ch=ch, nc=nc)
        self.desired_classes = desired_classes
        self.args = args if args is not None else IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.verbose = verbose  # Store verbosity preference
        self.init_criterion()

    def init_criterion(self):
        # Initialize the custom loss function
        self.loss = CustomDetectionLoss(self, self.desired_classes)