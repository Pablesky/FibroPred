import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import config as cfg

class DatasetFibro(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep = cfg.SPLIT_CHAR)
        
        self.data = self.data.drop(cfg.DELETE_COLUMNS, axis = 1)
        self.features_name = self.data.columns.difference(cfg.TARGET_COLUMNS)
        self.num_features = len(self.features_name)
        
        self.target_names = cfg.TARGET_COLUMNS
        self.label_encoders_input = {}
        self.max_values = {}
        
        self.vars_per_feature = []
        
        self.init_label_encoders()
        self.init_target_encoders()
        self.ini_var_per_feature()
        
    def return_features_name(self):
        return self.features_name
        
    def __len__(self):
        return len(self.data)
    
    def init_label_encoders(self):
        for feature in self.features_name:
            if feature not in cfg.SKIP_COLUMNS:
                self.label_encoders_input[feature] = LabelEncoder()
                self.label_encoders_input[feature].fit(self.data[feature])
            
            else:
                if feature not in self.max_values:
                    self.max_values[feature] = max(self.data[feature])
            
    def init_target_encoders(self):
        for target in self.target_names:    
            actual_column = self.data[target].fillna(-1.)
            self.data[target] = actual_column
                
    def return_encoders(self):
        return self.label_encoders_input
    
    def ini_var_per_feature(self):
        for feature in self.features_name:
            if feature not in cfg.SKIP_COLUMNS:
                self.vars_per_feature.append(len(self.label_encoders_input[feature].classes_))
            else:
                self.vars_per_feature.append(-1)
    
    def return_var_per_feature(self):
        return self.vars_per_feature
        
    def __getitem__(self, index):
        actual_row = self.data.iloc[index]
        
        output_dict = {}
        for feature in self.features_name:
            if feature not in cfg.SKIP_COLUMNS:
                output_dict[feature] = self.label_encoders_input[feature].transform([actual_row[feature]])
            else:
                output_dict[feature] = np.array([actual_row[feature] / self.max_values[feature]])
        
        for target in self.target_names:
            output_dict[target] = np.array([actual_row[target]])
            
        return output_dict
    

