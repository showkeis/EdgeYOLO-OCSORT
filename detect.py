from datetime import datetime as date
from loguru import logger
from association import *
from glob import glob
import cv2
import os
import numpy as np
from edgeyolo.utils import get_color
from edgeyolo.detect import Detector

"""=====================================================
Model			    Size	mAPval	mAPval	FPSAGX	Params
--------------------------------------------------------
EdgeYOLO-Tiny-LRELU	416	    33.1	50.5	206	    7.6 / 7.0
			        640	    37.8	56.7	109	
EdgeYOLO-Tiny		416	    37.2	55.4	136	    5.8 / 5.5
			        640	    41.4	60.4	67	
EdgeYOLO-S			640	    44.1	63.3	53	    9.9 / 9.3
EdgeYOLO-M			640	    47.5	66.6	46	    19.0 / 17.8
EdgeYOLO			640	    50.6	69.8	34	    41.2 / 40.5
========================================================"""

# All parameters are defined here as constants instead of using argparse
WEIGHTS = "edgeyolo_tiny_lrelu_coco.pth"
CONF_THRES = 0.25
NMS_THRES = 0.55
MP = False
FP16 = True
NO_FUSE = False
INPUT_SIZE = [416, 640]
SOURCE = "0" #"E:/videos/test.avi"
TRT = False
LEGACY = False
USE_DECODER = False
BATCH = 1
NO_LABEL = False
SAVE_DIR = "./output/detect/imgs/"
FPS = 99999
IS_GPU = True

if IS_GPU:
    import torch

def draw(imgs, results, class_names, line_thickness=3, draw_label=True):
    single = False
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        single = True
    out_imgs = []
    tf = max(line_thickness - 1, 1)
    for img, result in zip(imgs, results):
        # print(img.shape)
        if result is not None:
            # print(result.shape)
            for *xywh, obj, conf, cls in result:
                if not class_names[int(cls)] == "person":
                    continue
                c1 = (int(xywh[0]), int(xywh[1]))
                c2 = (int(xywh[2]), int(xywh[3]))
                color = get_color(int(cls), True)
                cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
                if draw_label:
                    label = f'{class_names[int(cls)]} {obj * conf:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # print(img.shape)
        out_imgs.append(img)
    return out_imgs[0] if single else out_imgs

def detect_single():
    import time
    exist_save_dir = os.path.isdir(SAVE_DIR)

    # detector setup
    detector = Detector
    detect = detector(
        weight_file=WEIGHTS,
        conf_thres=CONF_THRES,
        nms_thres=NMS_THRES,
        input_size=INPUT_SIZE,
        fuse=not NO_FUSE,
        fp16=FP16,
        use_decoder=USE_DECODER
    )

    # source loader setup
    if os.path.isdir(SOURCE):

        class DirCapture:

            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))

            def isOpened(self):
                return bool(len(self.imgs))

            def read(self):
                print(self.imgs[0])
                now_img = cv2.imread(self.imgs[0])
                self.imgs = self.imgs[1:]
                return now_img is not None, now_img

        source = DirCapture(SOURCE)
        delay = 0
    else:
        source = cv2.VideoCapture(int(SOURCE) if SOURCE.isdigit() else SOURCE)
        delay = 1

    all_dt = []
    dts_len = 300 // BATCH
    success = True

    # start inference
    count = 0
    t_start = time.time()
    while source.isOpened() and success:

        frames = []
        for _ in range(BATCH):
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < BATCH:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        results = detect(frames, LEGACY)
        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / BATCH:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / BATCH * 1000:.1f}ms", end="      ")

        key = -1

        # [print(result.shape) for result in results]

        imgs = draw(frames, results, detect.class_names, 2, draw_label=not NO_LABEL)
        # print([im.shape for im in frames])
        for img in imgs:
            # print(img.shape)
            cv2.imshow("EdgeYOLO result", img)
            count += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s"):
                if not exist_save_dir:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    exist_save_dir = True
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR, file_name), img)
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            cv2.destroyAllWindows()
            break

    logger.info(f"\ntotal frame: {count}, total average latency: {(time.time() - t_start) * 1000 / count - 1}ms")

if __name__ == '__main__':
    detect_single()