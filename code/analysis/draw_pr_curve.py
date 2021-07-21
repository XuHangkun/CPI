import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',default="../model/valid_info.csv")
parser.add_argument('--epoches',type=int,nargs="+",default=[1,2,3,4,5,6,7,8,9])
args = parser.parse_args()
print(args)

valid_info = pd.read_csv(args.input)
figure = plt.figure()
for i in range(len(args.epoches)):
    epoch = args.epoches[i]
    if "epoch_%d_true_y"%(epoch) not in valid_info.columns:
        continue
    y_true = valid_info["epoch_%d_true_y"%(epoch)].to_numpy()
    y_pred = valid_info["epoch_%d_pred_y"%(epoch)].to_numpy()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(recall,precision,label="epoch %d"%(epoch))
plt.xlabel("Recall",fontsize=14)
plt.xlim(0,1.1)
plt.ylabel("Precision",fontsize=14)
plt.ylim(0,1.1)
plt.legend()
plt.show()
