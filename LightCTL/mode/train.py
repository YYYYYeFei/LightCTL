import argparse
import pathlib
import time
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch import nn, optim

sys.path.append("..")
from data.dataloader import Dataset_independent,Dataset_independent1
from model.x_clip import Contractive_CNN_80_40 #LightCTL
from evaluate.index import MyConfusionMatrix


def train(args):
    print("\nStart read data...")
    root_data='../data/dataset/McPAS-TCR/'
    valid_root_data = '../data/dataset/YFV/'
    model_save_root=f"../result/checkpoint/"
    pathlib.Path(model_save_root).mkdir(parents=True, exist_ok=True)
    torch_dataset_train=Dataset_independent(root_data)
    trainloader = torch.utils.data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    torch_dataset_test=Dataset_independent1(valid_root_data)
    testloader = torch.utils.data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    model = Contractive_CNN_80_40(use_mlm=args.use_mlm, use_visual_ssl=args.use_visual_ssl).to(args.device)
    best_mix=0
    bestresult = 0
    acc_train_total = []
    acc_val_total = []
    loss_train_total = []
    loss_val_total = []
    for n in range(args.num_epoch):
        start = time.time()
        running_loss=0
        correct_train=0
        correct_test=0
        index_train=0
        index_val=0
        running_loss_test=0
        predict_prob=[]
        predict=[]
        lab_test=[]
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        print("\n[{}/{}] Start train data...".format(n,args.num_epoch))
        for SampleFeatureAuto_TCR,SampleFeatureAuto_pMHC,y in trainloader:
            model.train()
            y=(torch.squeeze(y, 1)).to(args.device)
            optimizer.zero_grad()
            result,loss_CL = model(SampleFeatureAuto_TCR.to(args.device),SampleFeatureAuto_pMHC.to(args.device))
            l = loss(result, y)+loss_CL
            running_loss = running_loss + l.item()
            l.backward()
            optimizer.step()
            _, predicted = torch.max(result.data, 1)
            correct_train += (predicted.long() == y).sum().item()
            index_train += len(y)
        acc_train_total.append(correct_train / index_train)
        loss_train_total.append(running_loss/index_train)
        print('correct_train:', correct_train)
        print('index_train:', index_train)
        print("train_Loss:{:.4f}".format(running_loss/index_train))
        print("train_acc:{:.4f}\n".format(correct_train / index_train))

        print("[{}/{}] Start test data...".format(n, args.num_epoch))
        model.eval()
        with torch.no_grad():
            for test_TCR,test_pMHC,test_y in testloader:
                test_y = torch.squeeze(test_y, 1).to(args.device)
                result_val,loss_CL_test = model(test_TCR.to(args.device),test_pMHC.to(args.device))
                _, predict_test = torch.max(result_val.data, 1)
                correct_test += (predict_test.long() == test_y).sum().item()
                index_val += len(test_y)
                l_test = loss(result_val, test_y)+loss_CL_test
                running_loss_test += l_test.item()
                predict_prob.extend(F.softmax(result_val, dim=1).data.cpu().squeeze(1)[:, 1])
                predict.extend(predict_test.cpu())
                lab_test.extend(test_y.cpu())

            result = MyConfusionMatrix(lab_test, predict)
            acc_val_total.append(result[0])
            loss_val_total.append(running_loss_test / index_val)
            fpr, tpr, thresholds = roc_curve(lab_test, predict_prob)
            roc_auc = auc(fpr, tpr)
            print("roc_auc:{:.4f}".format(roc_auc))

            if bestresult < roc_auc:
                bestresult = roc_auc
                best_mix=result
                print("the best AUC:", bestresult)
                print("the best confusion matirx:", best_mix)
                torch.save(model, model_save_root + '/bestmodel.pth')
                # torch.save(model.state_dict(), model_save_root+'/bestmodel.pth')

    print("\n验证集（{}）上的最佳结果".format(args.test_dataset_name))
    print("the best AUC:",bestresult)
    print("the best confusion matirx:", best_mix)


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
    parser.add_argument("--GPU_id", default=1, type=int,
                        help="GPU number",required=False)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.GPU_id)) if torch.cuda.is_available() else torch.device('cpu')
    # getattr(mode_arch, args.mode)(args)
    train(args)


if __name__ == '__main__':
    main()


"""
命令行说明：
python main.py --mode train --GPU_id 0
"""








