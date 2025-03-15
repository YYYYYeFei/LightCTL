"""
Description:we encode the sequence of TCR,antigen,and MHC based on antuencoder,and then
            but we didn't use thier final model instead of gradientboosting.
Negative selecting rules:the number of negative samples is the same as the postive instances.
Data:2022.10.11
Author:Fei Ye
"""
import argparse
import pathlib
import time
import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.append("..")
from data.dataloader import Dataset_independent,Dataset_independent1
from evaluate.index import MyConfusionMatrix


def test(args):
    print("Start read data...")
    root_data='../data/dataset/{}/'.format(args.test_data_name)
    model_load_root="../results/checkpoint/"
    model=torch.load(model_load_root + "bestmodel.pth", map_location=lambda storage, loc: storage.cuda(int(args.GPU_id)))

    torch_dataset_test=Dataset_independent1(root_data)
    testloader = torch.utils.data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
        drop_last=True
    )
    # model.load_state_dict(torch.load(model_load_root+'/bestmodel.pth'))
    correct_test=0
    index_val=0
    predict_prob=[]
    predict=[]
    lab_test=[]
    model.eval()
    for test_TCR,test_pMHC,test_y in testloader:
        test_y = torch.squeeze(test_y, 1).to(args.device)
        result_val,loss_CL_test = model(test_TCR.to(args.device),test_pMHC.to(args.device))
        _, predict_test = torch.max(result_val.data, 1)  # 输出每一行的最大值，以及最大值所在的索引,输出值为1 or 0
        correct_test += (predict_test.long() == test_y).sum().item()
        index_val += len(test_y)

        predict_prob.extend(F.softmax(result_val, dim=1).data.cpu().squeeze()[:, 1])  # 一次实验的所有概率值
        predict.extend(predict_test.cpu())  # 一次实验的预测输出标签
        lab_test.extend(test_y.cpu())  # 对应的实际值

    result = MyConfusionMatrix(lab_test, predict)
    fpr, tpr, thresholds = roc_curve(lab_test, predict_prob)
    roc_auc = auc(fpr, tpr)
    print("result:",result)
    print("roc_auc:{:.4f}".format(roc_auc))

    precision, recall, thersholds = precision_recall_curve(lab_test, predict_prob, pos_label=1)  # pos_label指定哪个标签为正样本
    AUPR = average_precision_score(lab_test, predict_prob, pos_label=1)  # 计算PR曲线下面积
    print("auc_pr:{:.4f}".format(AUPR))


    # subresult = []
    # subresult.extend(result)
    # subresult.extend([round(roc_auc, 4)])
    # # print(subresult)
    #
    # results = pd.DataFrame([subresult], columns=["Acc", "Sen", "Spec", "Prec", "MCC", "AUC"])
    #
    #
    # save_root = f"../result/independent_results/"
    # pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)
    # results.to_csv(save_root + "{}_metrics.csv".format(args.test_data_name), index=False)
    #
    # prec=pd.DataFrame(precision,columns=["precision"])
    # sen=pd.DataFrame(recall,columns=["recall"])
    # PR_result=pd.concat([prec, sen],axis=1)
    # PR_result.to_csv(save_root + "{}_PR.csv".format(args.test_data_name), index=False)
    #
    #
    # fpr=pd.DataFrame(fpr,columns=["fpr"])
    # tpr=pd.DataFrame(tpr,columns=["tpr"])
    # roc_result=pd.concat([fpr, tpr],axis=1)
    # roc_result.to_csv(save_root + "{}_roc.csv".format(args.test_data_name), index=False)
    #
    # label=pd.DataFrame(np.array(lab_test),columns=["label"])
    # predict=pd.DataFrame(np.array(predict),columns=["predict"])
    # predict_prob=pd.DataFrame(np.array(predict_prob),columns=["predict_prob"])
    # detail_result=pd.concat([label, predict,predict_prob],axis=1)
    # detail_result.to_csv(save_root + "{}_detail_results.csv".format(args.test_data_name), index=False)
    #
    # return result,fpr, tpr,roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--use_mlm", default=True,
                        help="Total batch size for training.")
    parser.add_argument("--use_visual_ssl", default=False,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epoch", default=150, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--test_data_name", default="covid", type=str,
                        help="The test dataset.")
    parser.add_argument("--GPU_id", default=1, type=int,
                        help="GPU number",required=False)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.GPU_id)) if torch.cuda.is_available() else torch.device('cpu')
    # getattr(mode_arch, args.mode)(args)
    test(args)


if __name__ == '__main__':
    main()



"""
命令行说明：
python main.py --mode test --test_data_name pMTnet_test --GPU_id 0
"""

