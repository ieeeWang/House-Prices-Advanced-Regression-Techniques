# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:40:04 2020
load saved SFS model stacking results and plot
@author: lwang
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle # Save a dictionary into a pickle file.

#%% load saved SFS model stacking results
# save it into a file named xxx.p
# pickle.dump(SFS_Stack_lasso_results, open("SFS_Stack_lasso_results.p", "wb"))  

# Load the dictionary back from the pickle file.
SFS_results = pickle.load(open("SFS_Stack_ridge_results.p", "rb"))
SFS_results1 = pickle.load(open("SFS_Stack_lasso_results.p", "rb"))
SFS_results2 = pickle.load(open("SFS_SVMpoly_Stack_ridge.p", "rb"))

# SFS_results:
# SFS_Stack_lasso_results = {'estimators_sel': estimators_sel,
#                            'RMSE_best': RMSE_best,
#                            'RMSE_all_steps': RMSE_all_steps,
#                            'elapsed_time':elapsed_time}

estimators_sel = SFS_results['estimators_sel']
RMSE_best = SFS_results['RMSE_best']
RMSE_all_steps = SFS_results['RMSE_all_steps']
elapsed_time = SFS_results['elapsed_time']


#%% plot SFS-stacking results
plt.figure(figsize=(10,6))
fs = 15
for i in range(len(RMSE_all_steps)): 
    plt.plot((i+1)*np.ones_like(RMSE_all_steps[i]), RMSE_all_steps[i], 'o',
             label=estimators_sel[i+1])
plt.plot(1+np.array(range(len(RMSE_best))), RMSE_best, '--')   
plt.grid() 
plt.legend(loc='best', fontsize=fs*0.8)
plt.xlabel("SFS steps", fontsize=fs)
plt.ylabel("RMSE on 10-fold CVs", fontsize=fs)
mytitle = ('Stacking model Sequential Forward Selection (SFS)'
           +'\n Evaluation in {:.1f} seconds on all steps').format(elapsed_time)
plt.title(mytitle, fontsize=fs)

#%% plot 3 group results in one
RMSE_best = SFS_results['RMSE_best']
RMSE_best1 = SFS_results1['RMSE_best']
RMSE_best2 = SFS_results2['RMSE_best']

plt.figure(figsize=(10,6))
fs = 15
plt.plot(1+np.array(range(len(RMSE_best))), RMSE_best, '-o', label= 'SFS-Lasso, Stack-Ridge')
plt.plot(1+np.array(range(len(RMSE_best1))), RMSE_best1, '-o', label= 'SFS-Lasso, Stack-Lasso')
plt.plot(1+np.array(range(len(RMSE_best2))), RMSE_best2, '-o', label= 'SFS-SVMploy, Stack-Ridge')
         
plt.grid() 
plt.legend(loc='best', fontsize=fs*0.8)
plt.xlabel("SFS steps", fontsize=fs)
plt.ylabel("RMSE on 10-fold CVs", fontsize=fs)
plt.title('Best stacking performance on each step of SFS', fontsize=fs)
