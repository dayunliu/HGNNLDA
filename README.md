# HGNNLDA
HGNNLDA is a computational framework which is used to predict lncRNA and drug sensitivity associations.It is described in detail in our paper "HGNNLDA: Predicting lncRNA-drug sensitivity associations via a dual channel hypergraph neural network".

## Requirements
* TensorFlow 1.15
* python 3.7
* numpy 1.19
* pandas 1.1
* scikit-learn 1.0
* scipy 1.5

## Data
We obtained lncRNA-drug sensitivity associations from the RNAactDrug database. RNAactDrug is a comprehensive database that provides drug sensitivity associated RNA molecules including lncRNA,miRNA,mRNAfrommulti-omicsdata.RNAactDrughas19,770 mRNAs, 11,119 lncRNAs, 438 miRNAs and 4,155 drugs. After removing redundant information, we constructed a benchmark dataset with 36,248 lncRNA-drug sensitivity associations, including 978 lncRNAs and 1,815 drugs.

## Run the demo

```bash
python main.py
```
