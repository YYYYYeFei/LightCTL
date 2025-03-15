import csv
import sys
from collections import Counter
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append("..")
from data.dataread import ReadMyCsv,readdata_to_array,readdata_to_np


class Dataset_independent(object):
    def __init__(self, root):
        self.root=root
        self.data_type = ["tcr", "hla", "antigen"]
        self.tcr_roottrain = self.root + "x_{}_{}_5x.npy".format('train', self.data_type[0])
        self.antigen_roottrain = self.root + "x_{}_{}_5x.npy".format('train', self.data_type[2])
        self.HLA_roottrain = self.root + "x_{}_{}_5x.npy".format('train', self.data_type[1])
        self.y_roottrain = self.root + "y_{}_5x.csv".format('train')

        self.tcr_roottest = self.root + "x_{}_{}_5x.npy".format('test', self.data_type[0])
        self.antigen_roottest = self.root + "x_{}_{}_5x.npy".format('test', self.data_type[2])
        self.HLA_roottest = self.root + "x_{}_{}_5x.npy".format('test', self.data_type[1])
        self.y_roottest = self.root + "y_{}_5x.csv".format('test')

        self.SampleFeatureAuto_TCRtrain = readdata_to_np(self.tcr_roottrain)  # list
        self.SampleFeatureAuto_antigentrain = readdata_to_np(self.antigen_roottrain)  # list
        self.SampleFeatureAuto_HLAtrain = readdata_to_np(self.HLA_roottrain)  # list
        self.ytrain = readdata_to_array(self.y_roottrain)  # list

        self.SampleFeatureAuto_TCRtest = readdata_to_np(self.tcr_roottest)  # list
        self.SampleFeatureAuto_antigentest = readdata_to_np(self.antigen_roottest)  # list
        self.SampleFeatureAuto_HLAtest = readdata_to_np(self.HLA_roottest)  # list
        self.ytest = readdata_to_array(self.y_roottest)  # list
        # print(self.SampleFeatureAuto_TCRtrain.shape)
        self.SampleFeatureAuto_TCR= np.concatenate((self.SampleFeatureAuto_TCRtrain,self.SampleFeatureAuto_TCRtest),axis=0)
        # print(self.SampleFeatureAuto_TCR.shape)
        self.SampleFeatureAuto_antigen = np.concatenate((self.SampleFeatureAuto_antigentrain,self.SampleFeatureAuto_antigentest),axis=0)
        self.SampleFeatureAuto_HLA = np.concatenate((self.SampleFeatureAuto_HLAtrain,self.SampleFeatureAuto_HLAtest),axis=0)
        self.y = np.concatenate((self.ytrain,self.ytest),axis=0)
        # print("样本量：",len(self.y))

    def __len__(self):
        return len(self.SampleFeatureAuto_HLA)

    def __getitem__(self, idx):
        SampleFeatureAuto_TCR=torch.FloatTensor(self.SampleFeatureAuto_TCR[idx])
        SampleFeatureAuto_antigen=torch.FloatTensor(self.SampleFeatureAuto_antigen[idx])
        SampleFeatureAuto_HLA=torch.FloatTensor(self.SampleFeatureAuto_HLA[idx])
        SampleFeatureAuto_pMHC=torch.unsqueeze(torch.cat((SampleFeatureAuto_antigen,SampleFeatureAuto_HLA),0),dim=0)
        y=torch.LongTensor(self.y[idx])
        return SampleFeatureAuto_TCR.permute(2,0,1),SampleFeatureAuto_pMHC,y

class Dataset_independent1(object):
    def __init__(self, root):
        self.root=root
        self.data_type = ["tcr", "hla", "antigen"]
        self.tcr_root = self.root + "x_{}_5x.npy".format(self.data_type[0])
        self.antigen_root = self.root + "x_{}_5x.npy".format(self.data_type[2])
        self.HLA_root = self.root + "x_{}_5x.npy".format(self.data_type[1])
        self.y_root = self.root + "y_5x.csv"

        self.SampleFeatureAuto_TCR = readdata_to_np(self.tcr_root)  # list
        self.SampleFeatureAuto_antigen= readdata_to_np(self.antigen_root)  # list
        self.SampleFeatureAuto_HLA = readdata_to_np(self.HLA_root)  # list
        self.y= readdata_to_array(self.y_root)  # list

    def __len__(self):
        return len(self.SampleFeatureAuto_HLA)

    def __getitem__(self, idx):
        SampleFeatureAuto_TCR=torch.FloatTensor(self.SampleFeatureAuto_TCR[idx])
        SampleFeatureAuto_antigen=torch.FloatTensor(self.SampleFeatureAuto_antigen[idx])
        SampleFeatureAuto_HLA=torch.FloatTensor(self.SampleFeatureAuto_HLA[idx])
        SampleFeatureAuto_pMHC=torch.unsqueeze(torch.cat((SampleFeatureAuto_antigen,SampleFeatureAuto_HLA),0),dim=0)
        y=torch.LongTensor(self.y[idx])
        return SampleFeatureAuto_TCR.permute(2,0,1),SampleFeatureAuto_pMHC,y