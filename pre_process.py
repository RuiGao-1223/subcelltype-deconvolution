import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import pandas as pd
import scipy
import numpy as np
import random
import os
import scanpy as sc
from scipy.special import gammaln
from scipy.stats import multinomial
import numpy as np
from scipy.optimize import minimize


def input_process(sc_m, sc_ct, bulk, makrer_list):
    ### 由sc_m和sc_ct生成sc_ref(normalized)
    ### 根据marker_list生成deconvolute的输入

    all_total_counts = sc_m.sum(axis=1).sum(0)
    ### 根据marker生成sc和bulk的特征矩阵
    sc_m=sc_m[makrer_list]
    bulk=bulk[makrer_list].T

    ### 计算每个ct的平均表达谱
    ### calculate probability mass for every celltype
    sc_all=sc_ct.merge(sc_m, left_index= True, right_index= True).set_index('cell_type')
    ct_list = sorted(list(set(sc_all.index)))
    sc_ref = []
    sum_ct=[]
    for ct_i in ct_list:
        sc_ct_i = sc_all[sc_all.index==ct_i]
        mean_ct_i = (sc_ct_i.mean(axis=0)).T
        sum_ct.append(sc_ct_i.sum(0).sum()) 
        sc_ref.append(mean_ct_i)
    sc_ref = pd.DataFrame(sc_ref,index=ct_list).T
    sc_ref_nor = sc_ref.apply(lambda col: col/col.sum(),axis=0)
    probability_mass = pd.DataFrame(sum_ct/all_total_counts,index=ct_list,columns=['probability_mass']).T
    return probability_mass,sc_ref_nor, bulk

def sc_correction(sc_ref, probability_mass):
    sc_ref_cor=[] 
    for cti in sc_ref.columns:
        sc_ref_cor.append(sc_ref[cti]*probability_mass[cti].values)
    return pd.DataFrame(sc_ref_cor).T