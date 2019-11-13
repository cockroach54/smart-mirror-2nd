import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
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
        # ****torch pretrained net****

        avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        flatten = Flatten()
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, 
                                                        pretrained_backbone=True)   
        self.backbone = net.backbone #256 

        # fc layer
        self.fc = nn.Sequential(
            avg_pool,
            flatten
        ) # 256
        
        # self.embed = nn.Sequential(
        #     self.backbone,
        #     self.fc
        # ) # 256
    
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.topk = 3
        self.threshold = 0.75
        self.feature_len = 256

        self.sort_order_descending = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.roi_upsample = nn.UpsamplingBilinear2d([224,224])
        self.roi_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    def embed(self, images):
        r = self(images)
        return self.fc(r)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        return self.backbone(images)[0]
    

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
        self.df_ref_online = pd.DataFrame(represented.cpu().numpy())
        self.df_ref_online['label'] = [reference_dataset.classes[i] for i in reference_dataset.targets]
        self.reference_means_online = self.df_ref_online.groupby('label').mean()
        self.dir_list_online = reference_dataset.classes

        # 즉각 임베딩 디비 생성
        self.setReferenceDataset(self.dir_list_online, self.df_ref_online, self.reference_means_online)
        return self.dir_list_online, self.df_ref_online, self.reference_means_online

    def addNewData_online(self, im_arr, label):
        """
        @params im_arr: h X w X 3 (RGB)
        @params label: "praL"
        """
        data = self.roi_transform(im_arr).unsqueeze(0)
        outputs = self.embed(data.to(self.device)).data
        new_feature = list(outputs.squeeze().cpu().numpy())
        new_feature.append(label)
        new_feature[-1]
        # concat to prev df_ref_online
        self.df_ref_online.loc[self.df_ref_online.shape[0]] = new_feature
        # revise reference_means_online 
        self.reference_means_online = self.df_ref_online.groupby('label').mean()
        # 즉각 임베딩 디비 생성
        self.setReferenceDataset(self.dir_list_online, self.df_ref_online, self.reference_means_online)
        print('[Detector]: addNewData_online -', label)
        return

    def addNewLabel_online(self, dirPath, label):   
        """
        @params dirPath: "server/vu-visor/static/images_ext"
        @params label: "praL"
        """
        finalDirPath = os.path.join(dirPath, label)
        ims = [Image.open(os.path.join(finalDirPath,i)) for i in os.listdir(finalDirPath)] # n_img X h X w X c
        ims_tensor = torch.stack([self.roi_transform(im) for im in ims])
        outputs = self.embed(ims_tensor.to(self.device)).data
        new_features =outputs.cpu().numpy() # n_img X 256

        new_df_ref_online = pd.DataFrame(new_features)
        new_df_ref_online['label'] = label
        # add new label
        if label not in self.dir_list_online: # 중복된 이름 없을 시 새로 등록
            self.dir_list_online.append(label)
            self.dir_list_online.sort() # 반드시 소팅해서 클래스 번호 재정렬 해줘야함
        # concat to prev df_ref_online
        _df_ref_online_non_overlap = self.df_ref_online[self.df_ref_online['label']!=label] # 중복된 이전 데이터 제거
        self.df_ref_online = pd.concat([_df_ref_online_non_overlap, new_df_ref_online], ignore_index=True)
        # revise reference_means_online 
        self.reference_means_online = self.df_ref_online.groupby('label').mean()
        # 즉각 임베딩 디비 생성
        self.setReferenceDataset(self.dir_list_online, self.df_ref_online, self.reference_means_online)
        print('[Detector]: addNewLabel_online -', label)
        return 
        
    # 임베딩 디비 생성
    def setReferenceDataset(self, sample_dir_list, df_ref_feature_sampled, reference_means_sampled):
        self.reference_classes = sample_dir_list
        self.reference_targets = list(df_ref_feature_sampled.iloc[:,-1])
        self.embedded_features_cpu = torch.tensor(df_ref_feature_sampled.iloc[:,:-1].as_matrix()).float().data # float64->float32(torch default), cpu
        self.embedded_features = self.embedded_features_cpu.to(self.device) # gpu
        self.embedded_means_numpy = reference_means_sampled.as_matrix()
        self.embedded_means = torch.tensor(self.embedded_means_numpy).float().data.to(self.device) # float64->float32(torch default), gpu

        self.c2i = {c:i for i,c in enumerate(self.reference_classes)}
        self.embedded_labels = torch.tensor([self.c2i[c] for c in self.reference_targets]).to(self.device)
        
        # make datafames for plot
        # ***필요없음 어차피 pca 해야함
        self.df_ = pd.DataFrame(np.array(self.embedded_features_cpu))
        self.df_['name'] = self.reference_targets
        self.centers_ = pd.DataFrame(self.embedded_means_numpy)
        self.centers_['name'] = self.reference_classes

    def fit_pca(self, n_components=4):
        # pca model fit
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components) # 256 차원 다쓰면 나중에 샘플링에서 계산 오류남, sample 개수보다 많으면 촐레스키 분해 에러나는듯
        self.transformed = self.pca.fit_transform(self.embedded_features_cpu)

        # show PCA features 
        self.df = pd.DataFrame(self.transformed)
        self.df['name'] = self.reference_targets
        self.centers = pd.DataFrame(self.pca.transform(self.embedded_means_numpy))
        self.centers['name'] = self.reference_classes

    def inference_tensor3(self, inputs, metric='cos', knn=True):
        """
        roi 피처맵을(nxcx7x7) 인풋으로 받음
        @params metric: [l2, mahalanobis, prob]
        @params inputs: # N x C x H x W
        """            
        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.fc(self.inputs).data # n_roi X features 256    

        # inference
        self.sort_order_descending = True         
        if knn: self.dists = my_cos(self.outputs, self.embedded_features)
        else: self.dists = my_cos(self.outputs, self.embedded_means)

        self.dists_sorted = self.dists.sort(dim=1, descending=self.sort_order_descending)
        if knn: self.predicts = self.embedded_labels[self.dists_sorted.indices][:,:self.topk]
        else: self.predicts = self.dists_sorted.indices[:,:self.topk]

        self.predicts_dist = self.dists_sorted.values[:,:self.topk]
        return self.predicts.data, self.predicts_dist.data    

    def inference_tensor4(self, inputs):
        """
        roi 피처맵을(nxcx7x7) 인풋으로 받음
        """            
        # inputs shape: Batch*C*H*W
        # input to backbone model
        self.inputs = inputs
        self.outputs = self.fc(self.inputs).data # n_roi X features 256    

        # inference
        self.dists = my_cos(self.outputs, self.embedded_means)
        self.predicts_dist = self.dists
        return self.predicts_dist.data    

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
        # imgPath = "./server/oracle_proj/predict.jpg"
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
    def save_plot(self, plotPath):
        self.fit_pca(n_components=4)
        _outputs = self.pca.transform(self.outputs.cpu())

        plt.figure(figsize=(9,9))
        plt.title("Vector space, pca_n: "+str(self.feature_len))
        ax = sns.scatterplot(x=0, y=1, hue='name', data=self.df, palette="Set1", legend="full", s=30)
        ax2 = sns.scatterplot(x=0, y=1, hue='name', data=self.centers, palette="Set1", s=150, legend=None, edgecolor='black')
        plt.scatter(_outputs[:,0], _outputs[:,1], marker='x', c='black', s=120)
        plt.savefig(plotPath)