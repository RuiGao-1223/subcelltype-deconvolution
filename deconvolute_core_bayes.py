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

def deconvolute_core_bayes(bulk,sc_ref):

    def calculate_posterior(a,sc_ref,bulk):
        # 计算每个基因在 bulk 样本中的表达预期值
        expected_counts = np.dot(sc_ref, a)
        expected_counts=expected_counts/expected_counts.sum()
        expected_counts[expected_counts < 1e-8] = 1e-8
        # 基因表达的总计数

        # 使用多项分布的概率质量函数计算观测到该 bulk 的概率
        observed_counts = np.array(bulk)  # 替换为实际的观测计数
        total_counts = np.sum(observed_counts)
        
        log_pmf = (
            gammaln(total_counts + 1) - 
            np.sum(gammaln(observed_counts + 1)) + 
            np.sum(observed_counts * np.log(expected_counts))
        )

        # 取指数来得到概率密度函数
        # pmf = np.exp(log_pmf)
        # print(prob)
        return log_pmf

    def optimize_pro_bayes(bulk_sample, sc_ref, a_init):
        # 超参数设置
        num_iterations = 50
        convergence_threshold = 1e-6  # 迭代停止的收敛阈值
        bounds = [(0, None)] * len(a_init)
        a_current = a_init.copy()
        consecutive_same_count = 0

        for iteration in range(num_iterations):
            result = minimize(
                fun=lambda a: -calculate_posterior(a, sc_ref, bulk_sample),
                x0=a_current,
                method='L-BFGS-B',
                bounds=bounds
            )
            a_optimized = result.x
            a_normalized = a_optimized / np.sum(a_optimized)  # 归一化参数
            # print(f"Iteration {iteration + 1}: Celltype Proportion = {a_normalized}")

            # 检查连续迭代结果是否相同
            if np.allclose(a_normalized, a_current, atol=convergence_threshold):
                consecutive_same_count += 1
                if consecutive_same_count == 5:  # 连续五次迭代结果相同，停止迭代
                    # print("Convergence achieved. Stopping iterations.")
                    break
            else:
                consecutive_same_count = 0
            # 更新当前参数作为下一次迭代的初始值
            a_current = a_normalized

        return a_current

    num_sample = bulk.shape[1]
    num_features = sc_ref.shape[1]
    initial_a_list = [np.ones(num_features) / num_features for _ in range(num_sample)]
    cell_pro = []
    for i in range(num_sample):
        a_init = initial_a_list[i]
        cell_pro.append(optimize_pro_bayes(bulk.iloc[:,i], sc_ref, a_init))
    cell_pro=pd.DataFrame(cell_pro,index=bulk.columns.values,columns=sc_ref.columns.values)
    return cell_pro