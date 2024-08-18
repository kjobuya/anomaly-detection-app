# import torch
from torch import stack, split, no_grad, cat, cuda, cdist, from_numpy
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from PIL import Image
import cv2
import os
from torchinfo import summary
import matplotlib.pyplot as plt
import scipy.ndimage
import datetime


class PatchCore:
    def __init__(self, neighbourhood_size, corset_subsample_size, batch_size=64, resize_shape=(500, 500)):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.feature_extractor = FeatureExtractor(resize_shape=resize_shape).to(self.device)
        self.neighbourhood_size = neighbourhood_size
        self.subsample_size = corset_subsample_size # percentage
        self.batch_size = batch_size
        
        self.memory_bank = None
        self.subsample_indices = None
        
        self.default_transform = transforms.Compose([
                    transforms.Resize(resize_shape),
                    transforms.ToTensor(),
        ])
        
    def transform_img(self, numpy_image: np.ndarray, transform=None):
        if transform is None:
            transform = self.default_transform
        pil_image = Image.fromarray(np.uint8(numpy_image))
        transformed_image = transform(pil_image)
        return transformed_image
    
    def apply_transforms_on_imgs(self, images):
        transformed_images = []
        for i, image in enumerate(images):
            trs_img = self.transform_img(image)
            transformed_images.append(trs_img)
            
        return transformed_images
    
    def neighbourhood_aggregation(self, features):
        pooling_layer = nn.AvgPool2d(kernel_size=self.neighbourhood_size, stride=1, padding=self.neighbourhood_size//2)
        output = pooling_layer(features)
        return output  

    def extract_features(self, images):
        self.feature_extractor.eval()
        features = []
        # covert list of images to batch
        images = stack(images)
        # split batch into sub-batches
        sub_batches  = split(images, self.batch_size)
        
        with no_grad():
            for i, sub_batch in enumerate(sub_batches): 
                # patches = self.extract_patches(image)
                sub_batch = sub_batch.to(self.device)
                patch_features = self.feature_extractor(sub_batch)
                features.append(patch_features)
        
        features = cat(features)          
        return features
    
    def build_memory_bank(self, normal_images, signal = None):
        normal_features = self.extract_features(normal_images)
        print("Finished extracting features")
        normal_features = self.neighbourhood_aggregation(normal_features)
        print("Finished neighbourhood aggregation")
        self.memory_bank = normal_features.view(normal_features.shape[0], -1).cpu().numpy()
        self.corset_subsampling()
        print("Finished corset subsampling")
        if signal is not None:
            signal.emit(True)
        return self.memory_bank
    
    def initialize_subset(self):
        centroid = np.mean(self.memory_bank, axis=0)[np.newaxis, :]
        # distances = euclidean_distances([centroid], self.memory_bank).flatten()
        
        # convert to tensors
        centroid_tensor, memory_bank_tensor = from_numpy(centroid).to(self.device), from_numpy(self.memory_bank).to(self.device)
        distances = cdist(centroid_tensor, memory_bank_tensor).flatten().cpu().numpy()
        
        farthest_point_index = np.argmax(distances)
        subset_indices = [farthest_point_index]
        return subset_indices
    
    def select_next_point(self, subset_indices):
        subset = self.memory_bank[subset_indices]
        subset_tensor, memory_bank_tensor = from_numpy(subset).to(self.device), from_numpy(self.memory_bank).to(self.device)
        
        # distances_to_subset = euclidean_distances(self.memory_bank, subset)
        distances_to_subset = cdist(subset_tensor, memory_bank_tensor).cpu().numpy()
        
        min_distances = np.min(distances_to_subset, axis=1)
        next_point_index = np.argmax(min_distances)
        return next_point_index
    
    def corset_subsampling(self):
        subset_indices = self.initialize_subset()
        size_limit = int(len(self.memory_bank) * self.subsample_size / 100)
        while len(subset_indices) < size_limit:
            next_point_index = self.select_next_point(subset_indices)
            subset_indices.append(next_point_index)
        self.subsample_indices = np.array(subset_indices)
    
    def detect_anomalies(self, test_images, signal = None):
        test_features = self.extract_features(test_images)
        test_features = self.neighbourhood_aggregation(test_features).cpu().numpy()
        subsampled_memory_bank = self.memory_bank[self.subsample_indices]
        
        reshaped_memory_bank = subsampled_memory_bank.reshape(subsampled_memory_bank.shape[0], test_features.shape[1], test_features.shape[2], test_features.shape[3])
        
        heatmap = np.zeros((test_features.shape[0],test_features.shape[2], test_features.shape[3]))
        for sample_idx in range(test_features.shape[0]):
            # build heatmap
            for i in range(test_features.shape[2]):
                for j in range(test_features.shape[3]):
                    tf = test_features[sample_idx, :, i, j]
                    tf_tensor = from_numpy(tf).to(self.device)[np.newaxis, :]
                    ssmb = reshaped_memory_bank[:, :, i, j]
                    ssmb_tensor = from_numpy(ssmb).to(self.device)
                    # distances_to_memory_bank = euclidean_distances([tf], ssmb)
                    distances_to_memory_bank = cdist(tf_tensor, ssmb_tensor).cpu().numpy()
                    
                    anomaly_score = np.min(distances_to_memory_bank, axis=1)
                    heatmap[sample_idx, i, j] = anomaly_score.item()
    
        
        if signal is not None:
            signal.emit(True)
            
        return heatmap

# Define the feature extractor using a pre-trained ResNet model
class FeatureExtractor(nn.Module):
    def __init__(self, resize_shape=(500, 500)):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        return_nodes = {
            # node_name: user-specified key for output dict
            # 'layer1.2.relu_2': 'layer1_output',
            'layer2.3.relu_2': 'layer2_output',
            # 'layer3.2.relu_2': 'layer3_output',
            # 'layer4.2.relu_2': 'layer4_output',
            }

        self.features = create_feature_extractor(resnet, return_nodes=return_nodes)
        
        # self.model_summary = summary(
        #     self.features, 
        #     input_size=(1, 3, resize_shape[0], resize_shape[1]),
        #     col_names=["input_size", "output_size", "num_params", "trainable"], 
        #     row_settings=["var_names"],
        #     )
        
    def forward(self, x):
        with no_grad():
            out = self.features(x)
            out = out['layer2_output']
            # out = out['layer3_output']
            # out = out.view(out.size(0), -1) # flattens
        return out
