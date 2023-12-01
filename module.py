import torch
import numpy as np
from common import DetectMultiBackend
from utils import non_max_suppression, scale_boxes, letterbox


def loadmodel(weights='last_half.engine', data='NG.yaml', imgsz=(1280, 1280)):
    model = DetectMultiBackend(weights, data=data)
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    return model


def loadimages(im0, img_size=(1280, 1280)):
    im = letterbox(im0, img_size)  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    return im, im0


def inference(model, im):
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im)
    return pred


def nms(pred, im, im0s, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    if len(im.shape) == 3:
        im = im[None]
    for _, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
    return pred[0].cpu().numpy().tolist()

