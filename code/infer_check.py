import pandas as pd
from inference import inference
import torch
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

small_data = pd.read_csv("../data/small_data.csv")
small_data = small_data[:20000]
small_data.columns = ["smiles","protein","label"]

true_labels = []
p_scores = []
p_positive_score = []
p_negative_score = []
loss = []
precision = []

def mean_seq(seq,k=30):
    result = []
    length = len(seq)//k
    for i in range(length-1):
        result.append(1.0*sum(seq[i*k:(i+1)*k])/k)
    return result

for i in tqdm(range(len(small_data))):
    smiles = small_data["smiles"][i]
    protein = small_data["protein"][i]
    t_label = small_data["label"][i]
    true_labels.append(t_label)
    p_score = inference(smiles,protein)
    p_scores.append(p_score)
    train_loss = binary_cross_entropy(torch.tensor(p_score), torch.tensor(t_label).float())
    loss.append(train_loss.item())
    if (p_score > 0.5) == t_label:
        precision.append(1)
    else:
        precision.append(0)
    if t_label:
        p_positive_score.append(p_score)
    else:
        p_negative_score.append(p_score)
fig_1 = plt.figure()
plt.plot(mean_seq(range(len(loss)),1000),mean_seq(loss,1000),label="loss")
plt.plot(mean_seq(range(len(loss)),1000),mean_seq(precision,1000),label="precision")
plt.xlabel("Item", fontsize=14)
fig_2 = plt.figure()
plt.hist(p_positive_score,histtype="step",hatch="/",bins=100,label="positive score")
plt.hist(p_negative_score,histtype="step",hatch="/",bins=100,label="netative score")
plt.xlabel("Score", fontsize=14)
plt.legend()
fig_3 = plt.figure()
fpr, tpr, thersholds = roc_curve(true_labels, p_scores, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
