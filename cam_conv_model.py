import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm_notebook as tqdm

class Flatten(torch.nn.Module):
    """
    torch.nn.Sequential에서 사용가능한 flatten 모듈
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.2):
        super().__init__()
        self.noise = torch.zeros(shape,shape).cuda()
        self.std   = std
        
    def forward(self, x):
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise
    
class CAM(nn.Module):
    def __init__(self, n_classes):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(CAM, self).__init__()
        self.scaler = MinMaxScaler()
        self.upsampler =  nn.UpsamplingBilinear2d([224,224])
        self.trans_normal = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#         ****torch pretrained net****
#         net = models.resnet50(pretrained=True)
#         modules = list(net.children())[:-2]      # delete all untill Conv_5 layer. - 2048
        avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # GAP
        flatten = Flatten()
#         self.backbone = nn.Sequential(*modules) #2048
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),   
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()          
        )
        
        self.class_len = n_classes
        self.classifier = nn.Linear(self.class_len, self.class_len, bias=False) #final linear classifier
        self.noise = DynamicGNoise(224, std=0.05)
        # Freeze resnet
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
        # 1x1 conv 
        self.conv_final = nn.Sequential(
            nn.Conv2d(256, self.class_len, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.class_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        # fc layer
        self.fc = nn.Sequential(
            avg_pool,
            flatten,
            self.classifier
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        # CNN backbone
        _images = self.noise(images)
        _conv = self.backbone(_images)
        featuremaps = self.conv_final(_conv)
        # fc layer
        logits = self.fc(featuremaps) 
        return logits, featuremaps
    
        
    def getCAM(self, images_normal):
        """
        get class activation map
        @param images_normal - (N x C x H x W) 
        @return cams_scaled - (N x 224 x 224) [0,1] weight matrix
        @return masks_np - (N x 224 x 224 x C) numpy [0,1] weight matrix
        """
        # forward to model
        logits, featuremaps = self(images_normal) # featuremaps (N x C x H x W)
        preds, indices = logits.softmax(dim=1).sort(dim=1, descending=True) # cam_weight (N x C)
        # cam_weights = self.classifier.weight[:,indices[:,0]].transpose(1,0)
        cam_weights = self.classifier.weight[:,[10]*len(indices)].transpose(1,0)

        # get class activation map
        cams = []
        for _featuremap, _cam_weight in zip(featuremaps, cam_weights):
            _cam = []
            for c,w in zip(_featuremap, _cam_weight): 
                _cam.append((c*w).data)
            _cam = torch.stack(_cam)
            cams.append(_cam)
        cams = torch.stack(cams)
            
        cams_updsampled = self.upsampler(cams).sum(dim=1) # batchx224x224
        # scale to [0,1] for recalibrartion weight
        cams_scaled = []
        for _cam_upsampled in cams_updsampled:
            _cam_scaled = self.scaler.fit_transform(_cam_upsampled.cpu().numpy())  
            cams_scaled.append(_cam_scaled)
        cams_scaled = torch.tensor(cams_scaled).data
        masks_np = np.stack([cams_scaled, cams_scaled, cams_scaled], axis=3)
        return cams_scaled, masks_np

