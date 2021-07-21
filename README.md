# **CPI**
## **Compound Protein Interation Prediction**
use deep learning to predict the possibility of compound and protein interaction.

## **Setup and dependencies**
- python 3.6
- pytorch >= 1.2.0
- numpy
- RDkit = 2019.03.3.0
- pandas
- Gensim >=3.4.0
- transformers
- tape_proteins

## **Dataset**
- small dataset

整理自中科院药物所发表的transformerCPI1算法的配套数据集，含有187,496个化合物-蛋白对（chemical-protein pair，CPP），蛋白质类型限制在G蛋白偶联受体（G protein-coupled receptor，GPCR）和激酶（kinase）两类，正负样本比例（active vs. inactive chemical-protein pairs）约为1：2

- large dataset

整理自ChEMBL2数据库，含有576,160个CPP，蛋白质类型不限，正负样本比例约为1:1

- detail

化合物结构序列用Simplified molecular-input line-entry system（SMILES）表示；蛋白质氨基酸序列用FASTA表示；相互作用活性基于BioAssay数据的IC50、EC50、Functional Assay等实验数据，经过标准阈值划分，从而判定化合物与蛋白之间的是否有相互作用。CPI符合以下条件的被判定为有相互作用：pIC50≥6.5、IC50≤300nM、pEC50≥6.5、EC50≤300nM。数据放在data目录下，文件名分别为small_data.csv和large_data.csv

## **Train and Test Model**
## Train the model
```bash
$ # you can find the the train scripts in directory scripts.
$ # if you want to use ChemBert or ProteinBert, you need to download the pretrained model first.
$ # I've put the link of the pretrained model at the directory ChemBERTa and prot_bert respectively.
$ python run_train.py --model_name [baseline,transformercpi] --evt_num EVT_NUM
```

## Test the model
```bash
$ # change the model at inference.py first
$ python run_test.py
``` 

## Draw the train info
```bash
$ # you can use the python file in analysis to analysis the train info 
$ # and evaluate the ROC,PRC,ACU of your trained model
$ python ./analysis/draw_train_info.py --input FILE_PATH
```

## **Author**
- 许杭锟(Xu Hangkun) [if you have problem, you cad add my wechat account MagicSci. I'am glad to communicate!]

## **Reference**
- Lifan Chen, Xiaoqin Tan, Dingyan Wang, Feisheng Zhong, Xiaohong Liu, Tianbiao Yang, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, Mingyue Zheng, TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments, Bioinformatics, Volume 36, Issue 16, 15 August 2020, Pages 4406–4414, https://doi.org/10.1093/bioinformatics/btaa524
- Evaluating Protein Transfer Learning with TAPE, https://github.com/songlab-cal/tape
- ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction, https://arxiv.org/abs/2010.09885
