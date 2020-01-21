import pandas as pd
import random
import numpy as np
import cardinality
from tqdm import tqdm_notebook

def bipartite_link(A, B, graph,score):
    sum_1=0
    sum_2=0
    sum_3=0

    for x in A:
        for y in B:
            a,b,c=score(x,y,graph)
            sum_1+=a
            sum_2+=b
            sum_3+=c
    return sum_1/(len(A)*len(B)),sum_2/(len(A)*len(B)),sum_3/(len(A)*len(B))


def commonneigh(a,b,graph):
    neib_a=set(list(graph.neighbors(a)))
    neib_b=set(list(graph.neighbors(b)))

    lis_2a=[]
    lis_2b=[]

    for x in list(neib_a):
        lis_2a+=(list(graph.neighbors(x)))

    for x in list(neib_b):
        lis_2b+=(list(graph.neighbors(x)))
    union_a=set(list(neib_b)+lis_2a)
    union_b=set(list(neib_a)+lis_2b)
    
    common_a=set(lis_2a).intersection(neib_b)
    common_b=set(lis_2b).intersection(neib_a)

    return cardinality.count(common_a),cardinality.count(common_b),(cardinality.count(common_a)+cardinality.count(common_b))/2

def jaccard(a,b,graph):

    neib_a=set(list(graph.neighbors(a)))
    neib_b=set(list(graph.neighbors(b)))

    lis_2a=[]
    lis_2b=[]

    for x in list(neib_a):
        lis_2a+=(list(graph.neighbors(x)))

    for x in list(neib_b):
        lis_2b+=(list(graph.neighbors(x)))
    union_a=set(list(neib_b)+lis_2a)
    union_b=set(list(neib_a)+lis_2b)
    
    common_a=set(lis_2a).intersection(neib_b)

    common_b=set(lis_2b).intersection(neib_a)
    
    try:
        jac_a=(cardinality.count(common_a))/(cardinality.count(union_a))
    except ZeroDivisionError:
        jac_a=0
    try:
        jac_b=(cardinality.count(common_b))/(cardinality.count(union_b))
    except ZeroDivisionError:
        jac_b=0

    return jac_a,jac_b,(jac_a+jac_b)/2

def admic_adar(a,b,graph):

    neib_a=set(list(graph.neighbors(a)))
    neib_b=set(list(graph.neighbors(b)))

    lis_2a=[]
    lis_2b=[]

    for x in list(neib_a):
        lis_2a+=(list(graph.neighbors(x)))

    for x in list(neib_b):
        lis_2b+=(list(graph.neighbors(x)))
    union_a=set(list(neib_b)+lis_2a)
    union_b=set(list(neib_a)+lis_2b)
    
    common_a=set(lis_2a).intersection(neib_b)
    common_a=graph.degree(list(common_a))
    #for i in list(common_a):

        
    common_b=set(lis_2b).intersection(neib_a)
    common_b=graph.degree(list(common_b))
    sum_a=0
    for i,j in common_a:
        if j!=0:
            sum_a+=1/(np.log(1+j))
    sum_b=0
    for i,j in common_b:
        if j!=0:
            sum_b+=1/(np.log(1+j))

    return sum_a,sum_b,(sum_a+sum_b)/2