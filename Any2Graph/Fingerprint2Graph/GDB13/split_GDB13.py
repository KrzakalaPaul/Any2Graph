import pandas as pd
import random
from sklearn.model_selection import train_test_split
random.seed(0)

path = 'data/GDB13_Subset-ABCDEFGH.smi'

df = pd.read_csv(path)

train_smiles,rest = train_test_split(df,test_size=0.1)
test_smiles,valid_smiles = train_test_split(rest,test_size=0.5)

train_smiles.to_csv('data/GDB13_train_smiles.csv',index=False)
test_smiles.to_csv('data/GDB13_test_smiles.csv',index=False)
valid_smiles.to_csv('data/GDB13_valid_smiles.csv',index=False)

