#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import csv
from collections import OrderedDict
__all__ = ["vis"]

vec_old = OrderedDict()
vec_new = OrderedDict()
norm = "0"
FULLVIEW = False
NORMFLAME = 10
STATUS = "NONE"
TRACKLINE = False
_c = None

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1,lineType=cv2.LINE_AA)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    global vec_old
    global vec_new

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        global vec_new
        global vec_old
        global norm
        global _c
        global STATUS

        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        intcenterbox = tuple(map(int, (x1, y1, x1 + w//2, y1 + h//2))) # center cordinate
        lowcenterbox = tuple(map(int, (x1 + w//2, y1 + h))) # center cordinate
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if not id_text in vec_new.keys():
            #vec_new[id_text] = [intcenterbox[2:4]] #tuple [(x,y)] #new register
            vec_new[id_text] = [lowcenterbox]
            #vec_old[id_text] = [intcenterbox[2:4]]
        else:
            #vec_old = vec_new
            #vec_new[id_text].append(intcenterbox[2:4]) # update new #tuple [(x1,y1),(x2,y2), ....]
            vec_new[id_text].append(lowcenterbox)

        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
        #cv2.circle(im,intcenterbox[2:4],9, color=color, thickness=1,lineType=cv2.LINE_AA)
        cv2.circle(im,lowcenterbox,9, color=color, thickness=1,lineType=cv2.LINE_AA)

        if TRACKLINE:
            if not len(vec_new[id_text]) == 1 or not len(vec_new[id_text]) == 0 :
                if FULLVIEW:
                    for id in vec_new.keys():
                        for t in range(len(vec_new[id])-1):
                            xy_1 = vec_new[id][t]
                            xy_2 = vec_new[id][t+1]
                            cv2.arrowedLine(im,pt1=xy_1,pt2=xy_2,color=color, thickness=2)
                        cv2.putText(im, id, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (30, 30, 255),
                                    thickness=text_thickness,lineType = cv2.LINE_AA)
                else:
                    for t in range(len(vec_new[id_text])-1):
                        xy_1 = vec_new[id_text][t]
                        xy_2 = vec_new[id_text][t+1]
                        cv2.arrowedLine(im,pt1=xy_1,pt2=xy_2,color=color, thickness=2)
                        cv2.circle(im,xy_2,5, color=color, thickness=1,lineType=cv2.LINE_AA)
                    if len(vec_new[id_text]) > NORMFLAME:
                        #print(id_text,id_text,frame_id)
                        norm = "{:.2f}".format(np.sqrt(np.abs(vec_new[id_text][-1][0] - vec_new[id_text][-NORMFLAME][0])**2+np.abs(vec_new[id_text][-1][1] - vec_new[id_text][-NORMFLAME][1])**2))
                        if float(norm) <= 2:
                            STATUS="STAY"
                            _c =  (255, 30, 30)
                        elif float(norm) > 2:
                            STATUS="MOVING"
                            _c =  (30, 30, 255)
                    #S = (intbox[0]+intbox[2])*(intbox[1]+intbox[3])*0.0000001
                    cv2.putText(im, id_text+" "+norm+" "+STATUS, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, _c,
                                thickness=text_thickness,lineType = cv2.LINE_AA)
        else:
            for i, tlwh in enumerate(tlwhs):
                x1, y1, w, h = tlwh
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                obj_id = int(obj_ids[i])
                id_text = '{}'.format(int(obj_id))
                if ids2 is not None:
                    id_text = id_text + ', {}'.format(int(ids2[i]))
                color = get_color(abs(obj_id))
                cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)
        #cv2.imshow('frame',im)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    with open('dct.csv', 'w', newline="") as f:  
        writer = csv.writer(f)
        for k, v in vec_new.items():
            writer.writerow([k, v])
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
