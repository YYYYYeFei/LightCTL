import math

def MyConfusionMatrix(y_real, y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    """
    分母为零的情况如何处理：

    """
    # 当除数不能为0的异常
    try:
        Acc = (TN + TP) / (TN + TP + FN + FP)
        print('Acc:%0.4f' % Acc)
    except ZeroDivisionError:
        Acc = 1000000
        print("Acc分母为零！！！")
    try:
        Sen = TP / (TP + FN)
        print('Sen:%0.4f' % Sen)
    except ZeroDivisionError:
        Sen = 1000000
        print("Sen分母为零！！！")
    try:
        Spec = TN / (TN + FP)
        print('Spec:%0.4f' % Spec)
    except ZeroDivisionError:
        Spec = 1000000
        print("Spec分母为零！！！")
    try:
        Prec = TP / (TP + FP)
        print('Prec:%0.4f' % Prec)
    except ZeroDivisionError:
        Prec = 1000000
        print("Prec分母为零！！！")
    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print('Mcc:%0.4f' % MCC)
    except ZeroDivisionError:
        MCC = 1000000
        print("MCC分母为零！！！")

    # 针对预测和真实预测完全相同，并且只有一种类别的情况。

    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))

    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:%0.4f'%(SumAcc / len(matrix)))
    print('AverageSen:%0.4f'%(SumSen / len(matrix)))
    print('AverageSpec:%0.4f'%(SumSpec / len(matrix)))
    print('AveragePrec:%0.4f'%(SumPrec / len(matrix)))
    print('AverageMcc:%0.4f'%(SumMcc / len(matrix)))
    return
