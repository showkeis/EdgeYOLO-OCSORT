from __future__ import print_function
from datetime import datetime as date
from loguru import logger
from association import *
from glob import glob
import cv2
import os
import time
import torch
import numpy as np
import random
from edgeyolo.detect import Detector
from collections import OrderedDict
from timer import Timer
from PIL import ImageFont, ImageDraw, Image
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression,BayesianRidge

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
WEIGHTS = "edgeyolo_coco.pth"
CONF_THRES = 0.25
NMS_THRES = 0.55
DET_THRES = 0.10
IOU_THRES = 0.1
MP = False
FP16 = True
NO_FUSE = False
INPUT_SIZE = [480, 640]
SOURCE = "0" #"yoru3_03-06-21-03-31.mp4" #"E:/videos/test.avi"
TRT = False
LEGACY = False
USE_DECODER = False
BATCH = 1
NO_LABEL = False
SAVE_DIR = "./output/detect/imgs/"
FPS = 99999
IS_GPU = True
TRACKING = True
TRACKLINE = True
FULLVIEW = False
TARGET_BOX = False
TARGETNUM = 0 # 67 cellphone
TRIMCENTER = False
RESIZE = True
vec_old = OrderedDict()
vec_new = OrderedDict()
norm = "0"
FULLVIEW = False
NORMFLAME = 10
STATUS = "NONE"
TRACKLINE = False
_c = None
prev_frame_time = None
FRAMENAME = "EDGEYOLO"

def get_color(id):
    # Set the seed for the random number generator
    random.seed(id)
    
    if id == 1:
        return (75, 0, 130)  # Indigo for ID 1
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color for other IDs

# Initialize outside of the function
mouse_pos = (0, 0)
left_click = False
_reg = None
selected_midpoints = []

def mouse_callback(event, x, y, flags, param):
    global mouse_pos, left_click
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        left_click = True

cv2.namedWindow(FRAMENAME)
cv2.setMouseCallback(FRAMENAME, mouse_callback)

def linear_reg(selected_midpoints):
    expanded_ls = [[j for sub in ((i,) if not isinstance(i, tuple) else i for i in inner) for j in sub] for inner in selected_midpoints]
    Xtrain = np.array([[j for i, j in enumerate(sub) if i not in {0, 11, 12}] for sub in expanded_ls]) #[data len, features]
    ytrain = np.array([sub[12] for sub in expanded_ls]) #(data len,)

    if Xtrain.shape[0] < 2:
        model = None
    else:
        # Create a Linear Regression model
        model = BayesianRidge()

        # Fit the model using the training data
        model.fit(Xtrain, ytrain)    
    return model

def predict_meter(_reg, point):
    # Assuming `point` is a list with 10 elements
    # Reshape it into a 2D array [[x1, y1, ..., x4, y4]] because sklearn requires 2D input
    expanded_ls = [[j for sub in ((i,) if not isinstance(i, tuple) else i for i in inner) for j in sub] for inner in point]
    Xtarget = np.array(expanded_ls).reshape(1, -1)
    print("X_for_Pred",Xtarget)
    # Now you can use the reshaped point in the prediction
    predicted_meter = _reg.predict(Xtarget)
    return predicted_meter


def plot_tracking(image, tlwhs, obj_ids, class_names=None, scores=None, frame_id=0, fps=0., ids2=None, results=None):
    global vec_new
    global left_click
    global selected_midpoints

    # Globalize the `_reg` variable
    global _reg

    meter_text = "None m"  # Add this line to ensure meter_text is always defined
    point = None

    # Initialize left_click if it's not defined
    if 'left_click' not in globals():
        left_click = False

    # Calculate fps
    global prev_frame_time
    cur_frame_time = time.time()
    if prev_frame_time is not None:
        fps = 1 / (cur_frame_time - prev_frame_time)
    prev_frame_time = cur_frame_time

    im = np.ascontiguousarray(np.copy(image[0]))
    print("im.shape :", im.shape)
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    corner_radius = 10  # 角の半径
    transparency = 0.1  # 透明度

    # Specify the shift amounts in the x and y direction
    shift_x = 5  # shift amount in the x-direction
    shift_y = 10  # shift amount in the y-direction

    # Convert image from OpenCV BGR format to PIL RGB format
    im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Specify the font .ttf file
    # Make sure the .ttf file is in your project directory or provide full path
    font = ImageFont.truetype("./font/YuGothB.ttc", int(10 * text_scale))

    # Draw the text on the image
    draw = ImageDraw.Draw(im_pil)
    draw.text((0, 0), 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)), font=font, fill=(255, 0, 40))

    # Convert back to OpenCV BGR format
    im = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        intcenterbox = tuple(map(int, (x1 + w // 2, y1 + h // 2)))  # center coordinate
        lowcenterbox = tuple(map(int, (x1 + w // 2, y1 + h // 2)))  # center coordinate
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if not id_text in vec_new.keys():
            vec_new[id_text] = []
        vec_new[id_text].append(lowcenterbox)
        if len(vec_new[id_text]) > 15:
            vec_new[id_text] = vec_new[id_text][-15:]  # 15点を保持するように修正

        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))

        if intbox[0] <= mouse_pos[0] <= intbox[2] and intbox[1] <= mouse_pos[1] <= intbox[3]:
            transparency = 0.4  # 透明度を上げる
            # 中心点やエッジの中点などの計算...

            midpoint_edge1 = ((intbox[0] + intbox[2]) // 2, intbox[1])
            midpoint_edge2 = (intbox[2], (intbox[1] + intbox[3]) // 2)
            midpoint_edge3 = ((intbox[0] + intbox[2]) // 2, intbox[3])
            midpoint_edge4 = (intbox[0], (intbox[1] + intbox[3]) // 2)
            center_point = ((intbox[0] + intbox[2]) // 2, (intbox[1] + intbox[3]) // 2)
            midpoints = [midpoint_edge1, midpoint_edge2, midpoint_edge3, midpoint_edge4, center_point]
            logger.info(f"Midpoints for box {i}: {midpoints}")
            for point in midpoints:
                cv2.circle(im, point, 2, (0,0,255), -1)  # Draw midpoints
        else:
            transparency = 0.1  # 透明度を元に戻す
            
        if left_click:
            try:
                #class_name = class_names[int(i)]
                print("point Debug",point)
                class_id = int(results[i][-1])
                # Assume `class_name` is the variable storing the class name
                entry = [i]+midpoints+[class_id]  # Temporary list to hold id, midpoint_edge1 x, midpoint_edge2 x, ..., center_point x, center_point y, class name
                input_meter = float(input())
                entry.append(input_meter)  # Add input meter(target value) to the temporary list
                selected_midpoints.append(entry)  # Save the list as a single entry in selected_midpoints
                print("ADD Save Point....",selected_midpoints)
                left_click = False  # Reset the click status

                _reg = linear_reg(selected_midpoints) # Param pred

                # Check if the model is defined
                if _reg is None:
                    print("Insufficient data points to train the model. Add more points.")
                else:
                    # Use the model
                    # TODO: Replace the placeholder line below with actual use of the model
                    print("Model trained successfully!")

            except Exception as e:
                print("Error: ", e)
                continue

        # Predict meter for all midpoints if the regression model is trained
        
        if _reg is not None:
            midpoint_edge1 = ((intbox[0] + intbox[2]) // 2, intbox[1])
            midpoint_edge2 = (intbox[2], (intbox[1] + intbox[3]) // 2)
            midpoint_edge3 = ((intbox[0] + intbox[2]) // 2, intbox[3])
            midpoint_edge4 = (intbox[0], (intbox[1] + intbox[3]) // 2)
            center_point = ((intbox[0] + intbox[2]) // 2, (intbox[1] + intbox[3]) // 2)
            midpoints = [midpoint_edge1, midpoint_edge2, midpoint_edge3, midpoint_edge4, center_point]
            #print("midpoint :",midpoints)
            print("------------------------------------------DebugZone")
            predicted_meter = predict_meter(_reg, midpoints) # return numpy format [ value ] scaler
            print("predicted_meter",predicted_meter)
            predicted_meter_fl = float(predicted_meter[0])
            if predicted_meter_fl is None:
                meter_text = "None m"
            else:
                meter_text = f"{predicted_meter_fl:.3f} m"
            # Display the meter text at each midpoint
            cv2.putText(im, meter_text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.ellipse(im, (intbox[0] + corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                    180, 0, 90, color, thickness=line_thickness)
        cv2.ellipse(im, (intbox[2] - corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                    270, 0, 90, color, thickness=line_thickness)
        cv2.ellipse(im, (intbox[0] + corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                    90, 0, 90, color, thickness=line_thickness)
        cv2.ellipse(im, (intbox[2] - corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                    0, 0, 90, color, thickness=line_thickness)

        # Draw the rounded rectangle
        intbox_poly = np.array([
            [intbox[0] + corner_radius, intbox[1]],
            [intbox[2] - corner_radius, intbox[1]],
            [intbox[2], intbox[1] + corner_radius],
            [intbox[2], intbox[3] - corner_radius],
            [intbox[2] - corner_radius, intbox[3]],
            [intbox[0] + corner_radius, intbox[3]],
            [intbox[0], intbox[3] - corner_radius],
            [intbox[0], intbox[1] + corner_radius],
        ])
        sub_img = im.copy()
        cv2.fillPoly(sub_img, [intbox_poly], color=color)
        im = cv2.addWeighted(sub_img, transparency, im, 1-transparency, 0)

        for point in vec_new[id_text]:
            cv2.circle(im, point[:2], 6, color=color, thickness=1,lineType=cv2.LINE_AA)
            cv2.circle(im, point[:2], 1, color=color, thickness=-1,lineType=cv2.LINE_AA) 

        # Convert image from OpenCV BGR format to PIL RGB format
        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        # Draw the text on the image
        draw = ImageDraw.Draw(im_pil)
        draw.text((intbox[0]+shift_x, intbox[1]+shift_y), id_text, font=font, fill=tuple([int(c) for c in color]))
        draw.text((intbox[0]+shift_x+20, intbox[1]+shift_y), meter_text, font=font, fill=tuple([int(c) for c in color]))

        # Convert back to OpenCV BGR format
        im = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow(FRAMENAME, im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if IS_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = False
else:
    device = torch.device("cpu")
    cpu = True

if TRACKING:
    from association import *
    from kalmanfilter import *
    from ocsort import *

# def draw_tracking(imgs, results, class_names, line_thickness=3, draw_label=True, online_ids=None):
#     single = False
#     if isinstance(imgs, np.ndarray):
#         imgs = [imgs]
#         single = True
#     out_imgs = []
#     tf = max(line_thickness - 1, 1)
#     for img, result in zip(imgs, results):
#         if result is not None:
#             for i, (*xywh, obj, conf, cls) in enumerate(result):
#                 if not class_names[int(cls)] == "person":
#                     continue
#                 c1 = (int(xywh[0]), int(xywh[1]))
#                 c2 = (int(xywh[2]), int(xywh[3]))
#                 color = get_color(int(cls), True)
#                 cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
#                 if draw_label:
#                     label = f'{class_names[int(cls)]} {obj * conf:.2f}'
#                     t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
#                     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#                     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#                     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#                 if online_ids is not None and i < len(online_ids):
#                     tid = int(online_ids[i])
#                     tid_label = f'ID: {tid}'
#                     tid_size = cv2.getTextSize(tid_label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
#                     tid_c2 = c1[0] + tid_size[0], c1[1] - tid_size[1] - t_size[1] - 6
#                     cv2.rectangle(img, c1, tid_c2, color, -1, cv2.LINE_AA)  # filled
#                     cv2.putText(img, tid_label, (c1[0], c1[1] - t_size[1] - 5), 0, line_thickness / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
#         out_imgs.append(img)
#     return out_imgs[0] if single else out_imgs

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

def trim_center(img, width, height):
    h, w = img.shape[:2]
    
    top = int((h / 2) - (height / 2))
    bottom = top+height
    left = int((w / 2) - (width / 2))
    right = left+width
    
    return img[top:bottom, left:right]

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
        cpu=cpu,
        fp16=FP16,
        use_decoder=USE_DECODER
    )

    if TRACKING == True:
        tracker = OCSort(det_thresh=DET_THRES, iou_threshold=IOU_THRES, use_byte=False)
        timer = Timer()
        frame_id = 0
        results = []

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

            if TRIMCENTER:
                frame = trim_center(frame, 640, 480)
            elif RESIZE:
                frame = cv2.resize(frame,tuple([INPUT_SIZE[1],INPUT_SIZE[0]]))
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

        results = detect(frames, LEGACY)[0] # torch [person number, 7] 7 ==> x1y1x2y2
        print("results : ",results)
        print("results type : ",type(results))
        # print("results shape : ",results.shape)

        if TARGET_BOX:
            if not results is None:
                indices = [i for i, result in enumerate(results) if result[-1] == TARGETNUM]
                results = results[indices]
            else:
                continue

        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / BATCH:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / BATCH * 1000:.1f}ms", end="      ")

        key = -1

        if TRACKING:
        # Initialize OCSort tracker

            online_targets = tracker.update(results, [INPUT_SIZE[0], INPUT_SIZE[1]], [frames[0].shape[0], frames[0].shape[1]])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                print(online_targets)
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

            # [print(result.shape) for result in results]
            print("online_ids : ",online_ids)
            #imgs = draw_tracking(frames, [results], detect.class_names, 2, draw_label=not NO_LABEL,online_ids=online_ids)
            timer.toc()
            print("online_targets shape :",online_targets.shape)
            imgs = plot_tracking(
                    frames, online_tlwhs, online_ids, detect.class_names, frame_id=frame_id + 1, fps=1. / timer.average_time, results=results
                )
            frame_id += 1

        else:
            imgs = draw(frames, [results], detect.class_names, 2, draw_label=not NO_LABEL)

            # print([im.shape for im in frames])
            for img in imgs:
                # print(img.shape)
                cv2.imshow(FRAMENAME, img)
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

    source.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_single()

"""-----------COCO Name----------------------------------
COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
"""