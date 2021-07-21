import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def mean_seq(seq,k=30):
    result = []
    length = len(seq)//k
    for i in range(length-1):
        result.append(1.0*sum(seq[i*k:(i+1)*k])/k)
    return result

def corr_train_valid(train_loss,valid_loss,k):
    """
    args:
        train_loss:
        valid_loss:
        k-gram    :
    """
    def cal_corr(seq_1,seq_2):
        # seq_1 = seq_1[:-1] - seq_1[1:]
        # seq_2 = seq_2[:-1] - seq_2[1:]
        corr = np.mean((seq_1-np.mean(seq_1))*(seq_2-np.mean(seq_2)))/(np.std(seq_1)*np.std(seq_2))
        return corr
    corr = []
    length = len(train_loss)//k
    for i in range(length-1):
        corr.append(
                cal_corr(train_loss[i*k:(i+1)*k],valid_loss[i*k:(i+1)*k])
        )
    return corr

parser = argparse.ArgumentParser()
parser.add_argument("--input",default="../model/train_info.csv")
parser.add_argument("--k",default=500,type=int)
parser.add_argument("--steps",default=2000000,type=int)
args = parser.parse_args()
data = pd.read_csv(args.input)
data = data[:args.steps]

# draw the loss and lr
loss_fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["train_loss"].to_numpy(),args.k),label="train loss")
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["valid_loss"].to_numpy(),args.k),label="valid loss")
ax1.legend()
#ax2.plot(mean_seq(data.index,args.k),mean_seq(data["lr"].to_numpy(),args.k),"g-",label="lr")
ax1.set_xlabel('Step',fontsize=14)
ax1.set_ylabel('Loss',fontsize=14)
#ax1.tick_params(colors="blue")
ax2.set_ylabel('lr', color='g',fontsize=14)
ax2.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
ax2.tick_params(colors="green")

# draw the precision and lr
precision_fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["train_precision"].to_numpy(),args.k),label="train precision")
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["valid_precision"].to_numpy(),args.k),label="valid precision")
ax1.legend()
#ax2.plot(mean_seq(data.index,args.k),mean_seq(data["lr"].to_numpy(),args.k),"g-",label="lr")
ax1.set_xlabel('Step',fontsize=14)
ax1.set_ylabel('Precision [%]',fontsize=14)
#ax1.tick_params(colors="blue")
ax2.set_ylabel('lr', color='g',fontsize=14)
ax2.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
ax2.tick_params(colors="green")
plt.show()
