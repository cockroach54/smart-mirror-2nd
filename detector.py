import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import cv2
import numpy as np
import pandas as pd
import os, shutil
from time import sleep
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
# import mymodels
# from mymodels.BN_Inception import BN_Inception

def my_cos(x,y):
    _x = F.normalize(x, p=2, dim=1)
    _y = F.normalize(y, p=2, dim=1)
    return _x.matmul(_y.transpose(0,1))

class Flatten(torch.nn.Module):
    """
    torch.nn.Sequential에서 사용가능한 flatten 모듈
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class OracleModel(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(OracleModel, self).__init__()
#         ****torch pretrained net****
#         net = models.shufflenetv2.shufflenet_v2_x1_0(pretrained=True) # 1024
        net = models.resnet50(pretrained=True)
#         net = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
#         net = models.mobilenet_v2(pretrained=True)
#         net = models.densenet201(pretrained=True)
#         self.backbone = net
        modules = list(net.children())[:-2]      # resnet conv_5
        avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        flatten = Flatten()
#         self.backbone = torch.nn.Sequential(*modules, avg_pool, flatten) #1280    
        self.backbone = torch.nn.Sequential(*modules) # 1024

# #         ****mobilenet****
#         self.BACKBONE_PATH = 'torch_models/mobile_ft.pt'
#         mobilenet = torch.load(self.BACKBONE_PATH)
#         modules = list(mobilenet.children())[:-1]
#         avg_pool = torch.nn.AvgPool2d(7, stride=1)
#         flatten = Flatten()
#         self.backbone = torch.nn.Sequential(*modules, avg_pool, flatten) #1280
        
# #         ****mobilnet distance matric learning****   
#         self.BACKBONE_PATH = 'mymodels/model_000400.pth'
#         self.backbone = torch.load(self.BACKBONE_PATH) #1280    

#         self.backbone = mymodels.create('Resnet50', dim=512, pretrained=True,
#                                         model_path='mymodels/ckp_ep210.pth.tar') #[227,227]->512

#         self.backbone = BN_Inception(dim=512, pretrained=True, 
#                                      model_path='mymodels/bn_inception-52deb4733.pth')

        # fc layer
        self.fc = nn.Sequential(
            avg_pool,
            flatten
        ) # 2048
        
        self.embed = nn.Sequential(
            self.backbone,
            self.fc
        ) # 2048
    
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.topk = 3
        self.threshold = 0.75
        self.feature_len = 2048
#         self.feature_len = 2048

        self.sort_order_descending = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.roi_upsample = nn.UpsamplingBilinear2d([224,224])
        self.roi_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    def forward(self, images):
        """Extract feature vectors from input images."""
        return self.backbone(images)[:,:self.feature_len]
    

    def makeAllReference_online(self, referenceImgDirPath):
        """
        임베딩 디비 생성... flask 서버용 
        utils.makeAllReferenceCSV 에서 csv 저장만 안함
        """
        reference_dataset = torchvision.datasets.ImageFolder(
            root=referenceImgDirPath,
            transform=transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        reference_loader = torch.utils.data.DataLoader(
            reference_dataset,
            batch_size=32,
            num_workers=0,
            shuffle=False
        )
        
        # get all data and input to model
        _temp = []
        for data, target in tqdm(reference_loader):
            outputs = self.embed(data.to(self.device)).data # .data 안하면 메모리 오버플로남...
            _temp.append(outputs)

        represented = torch.cat(_temp, dim=0)
        # raw data label 별 평균 구해두기
        df_ref = pd.DataFrame(represented.cpu().numpy())
        df_ref['label'] = [reference_dataset.classes[i] for i in reference_dataset.targets]
        reference_means = df_ref.groupby('label').mean()
        dir_list = reference_dataset.classes

        # 즉각 임베딩 디비 생성
        self.setReferenceDataset(dir_list, df_ref, reference_means)
        return dir_list, df_ref, reference_means
        
    # 임베딩 디비 생성
    def setReferenceDataset(self, sample_dir_list, df_ref_featere_sampled, reference_means_sampled):
        self.reference_classes = sample_dir_list
        self.reference_targets = list(df_ref_featere_sampled.iloc[:,-1])
        self.embedded_features_cpu = torch.tensor(df_ref_featere_sampled.iloc[:,:-1].as_matrix()).float().data # float64->float32(torch default), cpu
        self.embedded_features = self.embedded_features_cpu.to(self.device) # gpu
        self.embedded_means_numpy = reference_means_sampled.as_matrix()
        self.embedded_means = torch.tensor(self.embedded_means_numpy).float().data.to(self.device) # float64->float32(torch default), gpu

        self.c2i = {c:i for i,c in enumerate(self.reference_classes)}
        self.embedded_labels = np.array([self.c2i[c] for c in self.reference_targets])
        
        # make datafames for plot
        # ***필요없음 어차피 pca 해야함
        self.df_ = pd.DataFrame(np.array(self.embedded_features_cpu))
        self.df_['name'] = self.reference_targets
        self.centers_ = pd.DataFrame(self.embedded_means_numpy)
        self.centers_['name'] = self.reference_classes

    def fit_pca(self, n_components=4):
        # pca model fit
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components) # 2048 차원 다쓰면 나중에 샘플링에서 계산 오류남, sample 개수보다 많으면 촐레스키 분해 에러나는듯
        self.transformed = self.pca.fit_transform(self.embedded_features_cpu)

        # show PCA features 
        self.df = pd.DataFrame(self.transformed)
        self.df['name'] = self.reference_targets
        self.centers = pd.DataFrame(self.pca.transform(self.embedded_means_numpy))
        self.centers['name'] = self.reference_classes

    def inference_tensor2(self, inputs, metric='cos'):
        """
        @params metric: [l2, mahalanobis, prob]
        """
        # set metric function
        if metric == 'l2':
            self.metric_fn = self.calc_l2
            self.sort_order_descending = False
        elif metric == 'cos':
            self.metric_fn = self.calc_cos  
            self.sort_order_descending = True  

        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.embed(self.inputs).data # n_roi X features

        # inference
        if metric == 'l2':
            _diff = torch.stack([o-self.embedded_features for o in self.outputs]) # n_roi X classes X features
            self.dists = _diff.norm(dim=2, keepdim=True).squeeze()
        elif metric == 'cos':
            dists = [self.cos(self.embedded_features, o.unsqueeze(0)) for o in self.outputs]
            self.dists = torch.stack(dists)

        self.dists_sorted = self.dists.sort(dim=1, descending=self.sort_order_descending)
        self.predicts = torch.tensor(np.array([self.embedded_labels[idxs] for idxs in self.dists_sorted.indices.cpu()])).long()[:,:self.topk]
        self.predicts_dist = self.dists_sorted.values[:,:self.topk]
        return self.predicts.data, self.predicts_dist.data

    def inference_tensor(self, inputs, metric='cos'):
        """
        @params metric: [l2, mahalanobis, prob]
        """
        # set metric function
        if metric == 'l2':
            self.metric_fn = self.calc_l2
            self.sort_order_descending = False
        elif metric == 'cos':
            self.metric_fn = self.calc_cos  
            self.sort_order_descending = True  

        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.embed(self.inputs).data # n_roi X features

        # inference
        if metric == 'l2':
            _diff = torch.stack([o-self.embedded_means for o in self.outputs]) # n_roi X classes X features
            self.dists = _diff.norm(dim=2, keepdim=True).squeeze()
        elif metric == 'cos':
            dists = [self.cos(self.embedded_means, o.unsqueeze(0)) for o in self.outputs]
            self.dists = torch.stack(dists)

        self.dists_sorted = self.dists.sort(dim=1, descending=self.sort_order_descending)
        self.predicts = self.dists_sorted.indices[:,:self.topk]
        self.predicts_dist = self.dists_sorted.values[:,:self.topk]
        return self.predicts.data, self.predicts_dist.data

    def inference_tensor4(self, inputs, metric='cos'):
        """
        @params metric: [l2, mahalanobis, prob]
        """
        # set metric function
        if metric == 'l2':
            self.metric_fn = self.calc_l2
            self.sort_order_descending = False
        elif metric == 'cos':
            self.metric_fn = self.calc_cos  
            self.sort_order_descending = True  

        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.fc(self.inputs).data # n_roi X features

        # inference
        if metric == 'l2':
            _diff = torch.stack([o-self.embedded_features for o in self.outputs]) # n_roi X classes X features
            self.dists = _diff.norm(dim=2, keepdim=True).squeeze()
        elif metric == 'cos':
            dists = [self.cos(self.embedded_features, o.unsqueeze(0)) for o in self.outputs]
            self.dists = torch.stack(dists)

        self.dists_sorted = self.dists.sort(dim=1, descending=self.sort_order_descending)
        self.predicts = torch.tensor(np.array([self.embedded_labels[idxs] for idxs in self.dists_sorted.indices.cpu()])).long()[:,:self.topk]
        self.predicts_dist = self.dists_sorted.values[:,:self.topk]
        return self.predicts.data, self.predicts_dist.data

    def inference_tensor3(self, inputs, metric='cos', knn=True):
        """
        피처맵을 인풋으로 받음
        @params metric: [l2, mahalanobis, prob]
        @params inputs: # N x C x H x W
        """            
        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.fc(self.inputs).data # n_roi X features 1024    

        # inference
        self.sort_order_descending = True         
        if knn: self.dists = my_cos(self.outputs, self.embedded_features)
        else: self.dists = my_cos(self.outputs, self.embedded_means)

        self.dists_sorted = self.dists.sort(dim=1, descending=self.sort_order_descending)
        if knn: self.predicts = torch.tensor(self.embedded_labels[self.dists_sorted.indices.cpu()])[:,:self.topk]
        else: self.predicts = self.dists_sorted.indices[:,:self.topk]

        self.predicts_dist = self.dists_sorted.values[:,:self.topk]
        return self.predicts.data, self.predicts_dist.data    

    # 각 레이블 별 평균과의 L2 거리 계산 (PCA 적용안함)
    def calc_l2(self, label):
        mu = self.embedded_means[label]
        xx = self.outputs

        # 20*1280 X 1280*20 --diag--> 20
        # num_roi*dim X dim*num_roi --diag--> num_roi
        xx_sub_mu = xx.sub(mu)
        l2_dist = xx_sub_mu.matmul(xx_sub_mu.t()).sqrt().diag()
        return l2_dist.cpu()

    # 각 레이블 별 평균과의 cosine simility 계산 (PCA 적용안함)
    def calc_cos(self, label):
        mu = self.embedded_means[label]
        xx = self.outputs

        # 20*1280 X 1280*20 --diag--> 20
        # num_roi*dim X dim*num_roi --diag--> num_roi
        mu = mu.unsqueeze(0)
        cos_dist = self.cos(xx, mu)
        return cos_dist.cpu()        

    def inference_file(self, imgPath):              
        # imgPath = ".\\server\\oracle_proj\\predict.jpg"
        frame = cv2.imread(os.path.join(imgPath), cv2.IMREAD_COLOR)
        return self.inference_tensor(frame)

    def show_img(self, imgPath):
        frame = cv2.imread(os.path.join(imgPath), cv2.IMREAD_COLOR)
        # opencv frme input // H*W*C(BGR)
        # 0-255 3 channel
        inputs = torch.Tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))/255 # 여기 Tensor 소문자로 바꾸면 안됨... 차이 알아보기
        plt.imshow(inputs)
        plt.show()
        
    def show_tensor(self, t):
        plt.imshow(t.numpy().transpose([1,2,0]))
        plt.show()
        
    def plot(self):
        self.fit_pca(n_components=4)
        _outputs = self.pca.transform(self.outputs.cpu())

        plt.figure(figsize=(18,9))
        plt.title("Vector space, pca_n: "+str(self.feature_len))
        ax = sns.scatterplot(x=0, y=1, hue='name', data=self.df, palette="Set1", legend="full")
        ax2 = sns.scatterplot(x=0, y=1, hue='name', data=self.centers, palette="Set1", s=150, legend=None, edgecolor='black')
        plt.scatter(_outputs[:,0], _outputs[:,1], marker='x', c='black')
        plt.show()

    # save plot image for web
    def save_plot(self):
        self.fit_pca(n_components=4)
        _outputs = self.pca.transform(self.outputs.cpu())

        plt.figure(figsize=(9,9))
        plt.title("Vector space, pca_n: "+str(self.feature_len))
        ax = sns.scatterplot(x=0, y=1, hue='name', data=self.df, palette="Set1", legend="full", s=30)
        ax2 = sns.scatterplot(x=0, y=1, hue='name', data=self.centers, palette="Set1", s=150, legend=None, edgecolor='black')
        plt.scatter(_outputs[:,0], _outputs[:,1], marker='x', c='black', s=120)
        plt.savefig(os.path.join('.\\static', 'plot.jpg'))