import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from link_predictor import commonneigh, bipartite_link
from tqdm import tqdm_notebook

def evaluate(test_, G, score=commonneigh, silent = False):
    l=[]
    for i in tqdm_notebook(range(len(list(test_)))) if not silent else range(len(list(test_))):
        pr_1,pr_2,pr_3=bipartite_link(test_[i][0][0],test_[i][0][1],G,score)
        l.append((pr_1,pr_2,pr_3,test_[i][1]))
    l_1=pd.DataFrame(l,columns=["pr_1","pr_2","pr_3","real_"])
    d_1=l_1[["pr_1","real_"]]
    d_2=l_1[["pr_2","real_"]]
    d_3=l_1[["pr_3","real_"]]

    # d1=d_1.sort_values(by=["pr_1"],ascending=False)
    # d2=d_1.sort_values(by=["pr_1"],ascending=False)
    # d3=d_1.sort_values(by=["pr_1"],ascending=False)

    return (roc_auc_score(d_1[["real_"]].values,d_1[["pr_1"]].values),
            roc_auc_score(d_2[["real_"]].values,d_2[["pr_2"]].values),
            roc_auc_score(d_3[["real_"]].values,d_3[["pr_3"]].values))
