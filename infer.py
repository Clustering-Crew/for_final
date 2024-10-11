import os
import cv2
import argparse
import torch as t 
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import *

# Define function to display the image
def show_image(im, filename, name='image'):
    cv2.imwrite(f"./runs/{filename}.jpg", im)
    # cv2.imshow(name, im.astype(np.uint8))
    # cv2.waitKey(250)
    # cv2.destroyAllWindows()

# Define to show the boxes
def show_boxes(boxes_list, scores_list, labels_list, image, filename):
    thickness = 3
    height, width, _ = img.shape

    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(width * boxes_list[i][j][0])
            y1 = int(height * boxes_list[i][j][1])
            x2 = int(width * boxes_list[i][j][2])
            y2 = int(height * boxes_list[i][j][3])
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), int(thickness))
    
    show_image(image, filename)

# Define Parser
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
opt = parser.parse_args()

# YOLOv5-------------------------------------------------------------------------------------------------
v5_n = t.hub.load(
    "ultralytics/yolov5", 
    "custom",   path=r"D:\Underwater_imaging\All_results\combined\yolov5\combined_v5_n\weights\best.pt"
)

v5_s = t.hub.load(
    "ultralytics/yolov5", 
    "custom",   path=r"D:\Underwater_imaging\All_results\combined\yolov5\combined_v5_s\weights\best.pt"
)

# v5_m = t.hub.load(
#     "ultralytics/yolov5",
#     "custom",   path=r"D:\Underwater_imaging\All_results\combined\yolov5\combined_v5_m\weights\best.pt"
# )

# Set confidence
v5_n.conf = 0.25
v5_s.conf = 0.25
# v5_m.conf = 0.25
#----------------------------------------------------------------------------------------------------------

# YOLOv8-------------------------------------------------------------------------------------------------------

v8_n = YOLO(r"D:\Underwater_imaging\All_results\combined\yolov8\combined_v8_n\weights\best.pt")
v8_s = YOLO(r"D:\Underwater_imaging\All_results\combined\yolov8\combined_v8_s\weights\best.pt")
v8_m = YOLO(r"D:\Underwater_imaging\All_results\combined\yolov8\combined_v8_m\weights\best.pt")

#--------------------------------------------------------------------------------------------------------------

for filename in os.listdir(opt.source):
    path = os.path.join(opt.source, filename)
    img = cv2.imread(path)

    # Detections YOLOv5
    result_n = v5_n(img)
    result_s = v5_s(img)
    # result_m = v5_m(img)

    # Detection YOLOv8
    result_v8_n = v8_n.predict(source=img, conf=0.25)[0].boxes.cpu()
    result_v8_s = v8_s.predict(source=img, conf=0.25)[0].boxes.cpu()
    result_v8_m = v8_m.predict(source=img, conf=0.25)[0].boxes.cpu()

    v8_n_labels = result_v8_n.xyxyn.numpy()
    v8_n_scores = result_v8_n.conf.unsqueeze(dim=1).numpy()
    v8_n_classes = result_v8_n.cls.int().unsqueeze(dim=1).numpy()

    v8_s_labels = result_v8_s.xyxyn.numpy()
    v8_s_scores = result_v8_s.conf.unsqueeze(dim=1).numpy()
    v8_s_classes = result_v8_s.cls.int().unsqueeze(dim=1).numpy()

    v8_m_labels = result_v8_m.xyxyn.numpy()
    v8_m_scores = result_v8_m.conf.unsqueeze(dim=1).numpy()
    v8_m_classes = result_v8_m.cls.int().unsqueeze(dim=1).numpy()

    all_labels = []
    all_conf = []
    all_classes = []
    weights = [1, 1, 1, 1, 1, 1]

    yolov5_models = [result_n, result_s, result_m]

    for i in result_n.xyxyn:
        labels = []
        conf = []
        classes = []

        for ele in i: 
            labels.append(ele[:4].cpu().tolist())
            conf.append(ele[4].cpu().tolist())
            classes.append(ele[5].cpu().tolist())

        all_labels.append(labels)
        all_conf.append(conf)
        all_classes.append(classes)

    for i in result_s.xyxyn:
        labels = []
        conf = []
        classes = []

        for ele in i: 
            labels.append(ele[:4].cpu().tolist())
            conf.append(ele[4].cpu().tolist())
            classes.append(ele[5].cpu().tolist())

        all_labels.append(labels)
        all_conf.append(conf)
        all_classes.append(classes)

    # for i in result_m.xyxyn:
    #     labels = []
    #     conf = []
    #     classes = []
    #
    #     for ele in i:
    #         labels.append(ele[:4].cpu().tolist())
    #         conf.append(ele[4].cpu().tolist())
    #         classes.append(ele[5].cpu().tolist())
    #
    #     all_labels.append(labels)
    #     all_conf.append(conf)
    #     all_classes.append(classes)


    # Include results from YOLOv8
    all_labels.append(v8_n_labels)
    all_labels.append(v8_s_labels)
    all_labels.append(v8_m_labels)

    all_conf.append(v8_n_scores)
    all_conf.append(v8_s_scores)
    all_conf.append(v8_m_scores)

    all_classes.append(v8_n_classes)
    all_classes.append(v8_s_classes)
    all_classes.append(v8_m_classes)
    
    ## Weighted Box Fusion

    b, s ,l = weighted_boxes_fusion(all_labels, all_conf, all_classes, weights, iou_thr=0.25, skip_box_thr=0.0001)

    boxes_list = []
    scores_list = []
    labels_list = []
    boxes_list.append(b)
    labels_list.append(l)
    scores_list.append(s)

    show_boxes(boxes_list, scores_list, labels_list, img, filename)


### For Video

# vid = cv2.VideoCapture(opt.source)

# while(True):
#     ret, img = vid.read()

#     if ret:
    
#         result_n = v5_n(img)
#         result_s = v5_s(img)
#         result_m = v5_m(img)

#         all_labels = []
#         all_conf = []
#         all_classes = []
#         weights = [1, 2, 3]

#         for i in result_n.xyxyn:
#             labels = []
#             conf = []
#             classes = []

#             for ele in i: 
#                 labels.append(ele[:4].cpu().tolist())
#                 conf.append(ele[4].cpu().tolist())
#                 classes.append(ele[5].cpu().tolist())

#             all_labels.append(labels)
#             all_conf.append(conf)
#             all_classes.append(classes)

#         for i in result_s.xyxyn:
#             labels = []
#             conf = []
#             classes = []

#             for ele in i: 
#                 labels.append(ele[:4].cpu().tolist())
#                 conf.append(ele[4].cpu().tolist())
#                 classes.append(ele[5].cpu().tolist())

#             all_labels.append(labels)
#             all_conf.append(conf)
#             all_classes.append(classes)

#         for i in result_m.xyxyn:
#             labels = []
#             conf = []
#             classes = []

#             for ele in i: 
#                 labels.append(ele[:4].cpu().tolist())
#                 conf.append(ele[4].cpu().tolist())
#                 classes.append(ele[5].cpu().tolist())

#             all_labels.append(labels)
#             all_conf.append(conf)
#             all_classes.append(classes)

#         # print("Labels -\n")
#         # print(all_labels)
#         # print("#" * 20)
#         # print("confidences - \n")
#         # print(all_conf)
#         # print("#" * 20)
#         # print("classes - \n")
#         # print(all_classes)


#         b, s ,l = weighted_boxes_fusion(all_labels, all_conf, all_classes, weights, iou_thr=0.25, skip_box_thr=0.0001)

#         # print("################After Ensembling##############")

#         # print("################################Boxes################################")
#         # print(b)
#         # print("################################Scores################################")
#         # print(s)
#         # print("################################Labels################################")
#         # print(l)

#         boxes_list = []
#         scores_list = []
#         labels_list = []
#         boxes_list.append(b)
#         labels_list.append(l)
#         scores_list.append(s)
#         show_boxes(boxes_list, scores_list, labels_list, img)