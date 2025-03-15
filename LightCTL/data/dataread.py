import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
def ReadMyTxt(SaveList, fileName):
    csv_reader = csv.reader(open(fileName),delimiter='\t')
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return SaveList
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def readdata(path):
    SampleFeature = []
    feature = []
    ReadMyCsv(feature, path)
    for i in range(len(feature)):
        c = []
        for j in range(len(feature[0])):
            c.append(float(feature[i][j]))
        SampleFeature .append(c)
    return SampleFeature
def readdata_to_np(path):
    feature=np.load(path)
    # print(feature.shape)
    # SampleFeature = []
    # feature = []
    # ReadMyCsv(feature, path)
    # for i in range(len(feature)):
    #     c = []
    #     for j in range(len(feature[0])):
    #         c.append(float(feature[i][j]))
    #     SampleFeature .append(c)
    # SampleFeature = np.array(SampleFeature)
    return feature
def readdata_to_array(path):
    SampleFeature = []
    feature = []
    ReadMyCsv(feature, path)
    for i in range(len(feature)):
        c = []
        for j in range(len(feature[0])):
            c.append(float(feature[i][j]))
        SampleFeature .append(c)
    SampleFeature = np.array(SampleFeature)
    return SampleFeature
def labelgeneration(num_pos,num_neg):
    #samplelabel
    # SampleLabel
    SampleLabel = []
    counter = 0
    while counter < num_pos:
        SampleLabel.append(1)
        counter = counter + 1
    counter1 = 0
    while counter1 < num_neg:
        SampleLabel.append(0)
        counter1 = counter1 + 1
    print("len(SampleLabel):{}".format(len(SampleLabel)))
    return SampleLabel
def train_test_dataset(SampleFeatureAuto_TCR,SampleFeatureAuto_antigen,SampleFeatureAuto_HLA,SampleLabel):
    # 打乱数据集顺序
    counter = 0
    R = []
    while counter < len(SampleFeatureAuto_TCR):
        R.append(counter)
        counter = counter + 1
    random.shuffle(R)

    RSampleFeature = []
    RSampleLabel = []
    counter = 0
    while counter < len(SampleFeatureAuto_TCR):
        RSampleFeature.append(SampleFeatureAuto_TCR[R[counter]]+SampleFeatureAuto_antigen[R[counter]]+SampleFeatureAuto_HLA[R[counter]])
        RSampleLabel.append(SampleLabel[R[counter]])
        counter = counter + 1

    print('len(RSampleFeature)', len(RSampleFeature))
    print('len(RSampleLabel)', len(RSampleLabel))
    SampleFeature = []
    SampleLabel = []
    SampleFeature = RSampleFeature
    SampleLabel = RSampleLabel
    X = np.array(SampleFeature)
    y = np.array(SampleLabel)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
def fusion_dataset(SampleFeatureAuto_TCR,SampleFeatureAuto_antigen,\
                           SampleFeatureAuto_HLA):
    counter=0
    SampleFeature=[]
    while counter < len(SampleFeatureAuto_TCR):
        SampleFeature.append(SampleFeatureAuto_TCR[counter]+SampleFeatureAuto_antigen[counter]+SampleFeatureAuto_HLA[counter])
        counter = counter + 1

    return SampleFeature