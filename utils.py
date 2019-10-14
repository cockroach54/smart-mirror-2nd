import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import pandas as pd
import os, shutil, cv2
from time import sleep
import subprocess
from tqdm import tqdm

# Calc IOU
def calcIOU(box1, box2):
    area_box1 = box1[2]*box1[3]
    area_box2 = box2[2]*box2[3]
    x1_max = max(box1[0], box2[0])
    x2_min = min(box1[0]+box1[2], box2[0]+box2[2])
    y1_max = max(box1[1], box2[1])
    y2_min = min(box1[1]+box1[3], box2[1]+box2[3])
    
    area_intersection = max(0, x2_min-x1_max) * max(0, y2_min-y1_max)
    area_union = area_box1+area_box2-area_intersection +1e-9
    return area_intersection/area_union

def non_max_sup_one_class(bboxes, threshold=0.2, descending=False):
    """
    @params threshold - 
    @params ascending - 기본이 내림차순,
    """
    bboxes = list(bboxes)
    bboxes.sort(key = lambda x: x[2], reverse=descending) # 거리값이므로 오름차순, 확률이면 내림차순  
    bboxes = np.array(bboxes)
    keeps = [True]*len(bboxes)

    for i, bbox in enumerate(bboxes):
        if not keeps[i]: continue
        for j in range(i+1, len(bboxes)):
            if not keeps[i]: continue
            iou_res = calcIOU(bbox[0], bboxes[j][0])
            if iou_res>threshold: keeps[j] = False
    return bboxes[keeps]

edge_model = "./model.yml.gz"
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_model)

def rpn(im_opencv, num_boxs, scale=1):
    """
    region proposal network
    """
    global edge_detection
    
    def makeEdgeBox(scale):
        im_opencv_scaled = cv2.resize(im_opencv, (int(im_opencv.shape[1]*scale), int(im_opencv.shape[0]*scale)), 
                                      interpolation=cv2.INTER_CUBIC).astype(np.float32)

        edges = edge_detection.detectEdges(im_opencv_scaled / 255.0)

        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(num_boxs)
        boxes = edge_boxes.getBoundingBoxes(edges, orimap)
        if type(boxes)==tuple: boxes=boxes[0] # 버전에따라 다르게 나오는거 보정
        return boxes, im_opencv_scaled
    
    (boxes, im_opencv_scaled) = makeEdgeBox(scale)
    # bbox 하나도 없으면 전체샷이라도 저장
    if len(boxes)==0: boxes=np.array([[0,0,im_opencv_scaled.shape[1],im_opencv_scaled.shape[0]]])
    boxes = (boxes/scale).round().astype(np.int)
    # 박스 개수 절반보다 모자라면 스케일 키워서 한번더
    if len(boxes)<(num_boxs/2):
        scale = scale*2
        (_boxes, im_opencv_scaled) = makeEdgeBox(scale)
        if len(_boxes)>0:
            _boxes = (_boxes/scale).round().astype(np.int)
            boxes = np.concatenate([boxes,_boxes])[:num_boxs] # 이전 스케일의 박스와 concat

    return boxes

def rpn2(im, n_slice_x, n_slice_y, scale=(1,1)):
    """
    n분할 rpn
    @params im: nomalized image tensor N x C x W x H
    @return 
    """
    len_y, len_x, _ = im.shape
    w = int(len_x/n_slice_x)
    h = int(len_y/n_slice_y)    

    cxs = [int(w/2)+w*i for i in range(n_slice_x)]
    cys = [int(h/2)+h*i for i in range(n_slice_y)]
    
    rois = []
    boxes = []
    for cx in cxs:
        for cy in cys:
            x=int(cx-w/2); y=int(cy-h/2)
            w_diff = w*(1-scale[0])/2
            h_diff = h*(1-scale[1])/2
            w_modi = int(w*scale[0])
            h_modi = int(h*scale[1])
            x_modi = int(x+w_diff)
            y_modi = int(y+h_diff)

            boxes.append([max(0,x_modi),max(0,y_modi),w_modi,h_modi]) # x,y,w,h

    boxes = np.array(boxes)
    return boxes

def get_rois(images, featuremaps, bboxes):
    # calc bbox ratio
    ratio_y = featuremaps.shape[2]/images.shape[2]
    ratio_x = featuremaps.shape[3]/images.shape[3]
    bboxes_scaled = bboxes.clone()#.detach()

    bboxes_scaled[:,0] = bboxes_scaled[:,0]*ratio_x # for x
    bboxes_scaled[:,1] = bboxes_scaled[:,1]*ratio_y # for y
    bboxes_scaled[:,2] = bboxes_scaled[:,2]*ratio_x # for w
    bboxes_scaled[:,3] = bboxes_scaled[:,3]*ratio_y # for h

    # x,y,w,h -> x1, y1, x2, y2 그래디언트 학습되는 변수가 아니므로 inplace 계산 들어가도 괜찮다
    bboxes_scaled[:, 2] = bboxes_scaled[:, 0] + bboxes_scaled[:, 2]
    bboxes_scaled[:, 3] = bboxes_scaled[:, 1] + bboxes_scaled[:, 3]
    crops = torchvision.ops.roi_align(featuremaps, [bboxes_scaled], [7,7])
    return crops

class Flatten(torch.nn.Module):
    """
    torch.nn.Sequential에서 사용가능한 flatten 모듈
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class UBBR(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(UBBR, self).__init__()
#         ****torch pretrained net****
        net = models.resnet50(pretrained=True)
#         modules = list(net.children())[:-3]      # delete all untill Conv_4 layer. - 1024
        modules = list(net.children())[:-2]      # delete all untill Conv_5 layer. - 2048
        avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        flatten = Flatten()
        self.backbone = torch.nn.Sequential(*modules) #1024
#         self.backbone = torch.nn.Sequential(*modules, avg_pool, flatten) #1024
    
        # RoIAlign layer
#         self.roi_align = RoIAlign(7, 7) #.to('cuda')
#         self.roi_upsample = nn.UpsamplingBilinear2d([7,7]) # roi-align 대용
        
        # fc layer
        self.fc = nn.Sequential(
#             nn.Conv2d(1024, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
#             nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            flatten,
            nn.ReLU(),
            nn.Linear(7*7*16, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 4, bias=True),
        )
        
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.roi_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        
    def forward(self, images, boxes):
        """Extract feature vectors from input images."""
        # CNN backbone
        featuremaps = self.backbone(images)
        
        crops = get_rois(images, featuremaps, boxes)
        
        # fc layer
        offsets = self.fc(crops) 
        return offsets
    
def regression_transform(bboxes, offsets):
    """
    clac bboxes_adj using offsets
    @params bboxes_adj - N x 4 float tensor
    @params offsets - N x 4 float tensor
    """
    bboxes_adj = bboxes.clone().detach()
    # w,h -> x,y 
    # w,h 값이 x,y계산에 들어가므로 계산순서 바뀌면 안됨
    bboxes[:,2] = bboxes_adj[:,2]/(offsets[:,2].exp()) # w
    _w = bboxes[:,2].clone().detach() # backprop inplace error 방지!!!!!!!!! 중요
    bboxes[:,3] = bboxes_adj[:,3]/(offsets[:,3].exp()) # h
    _h = bboxes[:,3].clone().detach()
    bboxes[:,0] = bboxes_adj[:,0] - offsets[:,0]*_w # x
    bboxes[:,1] = bboxes_adj[:,1] - offsets[:,1]*_h  # y
    
    return bboxes    

def box_cvt(box):
    """
    auto grad possable
    @params box - N x 4
    x,y,w,h --> x1,y1,x2,y2
    """
    box2 = box.clone().float()
    box2[:,0] = box[:,0]
    box2[:,1] = box[:,1]
    box2[:,2] = box[:,0] + box[:,2]
    box2[:,3] = box[:,1] + box[:,3]
    return box2