# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:19:09 2020
SFS-model stacking (v3)
@author: lwang
"""
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Save a dictionary into a pickle file
import pickle
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,Lasso,LassoCV
from sklearn.experimental import enable_hist_gradient_boosting #need by below
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# helper functions below
from utils import plot_regression_results, show_CV_performance
from utils import plot_lasso_path, get_best_score, SFS_stack_models

#%% global settingRidgeCV
mse = 'neg_mean_squared_error'
nfold = 10 # of CV fold

#%% 3.1 Load preprocessed data - the number of catgarical columns remain
df_train = pd.read_csv("./data/train_3A.csv")
df_test  = pd.read_csv("./data/test_3A.csv")
# df_train = pd.read_csv("./data/train_3B.csv")
# df_test  = pd.read_csv("./data/test_3B.csv")

print('train:',df_train.shape)
print('test:',df_test.shape)

# extract the whole features for ML
X = df_train[df_train.columns[:-1]].copy()
y = df_train["SalePrice_Log"].copy()
final_test = df_test.copy() # final test X

#%% Models with optimal params determined by 10-folds CV
lasso_opt = Lasso(alpha = 0.00423)
ridge_opt = Ridge(alpha = 1)
knn_opt = KNeighborsRegressor(n_neighbors = 10, weights= 'distance')
rf = RandomForestRegressor(random_state=2) # use default params: N=100
boost = HistGradientBoostingRegressor() # default 'max_iter': 100
dt = DecisionTreeRegressor(random_state=5) # use default params
SVMrbf_opt = SVR(kernel='rbf', C= 1.0, gamma = 0.01)
SVMli = SVR(kernel='linear')
SVMpoly_opt = SVR(kernel='poly',degree = 3)

estimators_list = [('Lasso', lasso_opt),
                    ('Ridge', ridge_opt),
                    ('KNN', knn_opt),
                    ('RF', rf),
                    ('DT', dt),
                    ('Boost', boost),
                    ('SVMrbf', SVMrbf_opt),
                    ('SVMli', SVMli),
                    ('SVMpoly', SVMpoly_opt)]


#%% do SFS-model stacking        
start_time = time.time()
estimators_sel, RMSE_best, RMSE_all_steps = SFS_stack_models(estimators_list, 
                                                              X, y, nfold=10)
elapsed_time = time.time() - start_time
print('elapsed_time:', elapsed_time)

# create a dictionary to save
SFS_Stack_results = {'estimators_sel': estimators_sel,
                            'RMSE_best': RMSE_best,
                            'RMSE_all_steps': RMSE_all_steps,
                            'elapsed_time':elapsed_time}

# save it into a file named xxx.p
pickle.dump(SFS_Stack_results, open("SFS-laso_Stack-SVMpoly_results.p", "wb"))  

# plot SFS-stacking results
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


#%% choose which models to stacking
# All base models stacking:
# (1) lasso-final stacking = 0.12349
# (2) ridge-final stacking = 0.12436
# estimators = estimators_list 

# 4 base models for ridge-final stacking model, = 0.12296 / 0.12264
estimators = [estimators_list[0], estimators_list[5],
              estimators_list[6],estimators_list[7]]

# 4 base models for lasso-final stacking model, = 0.12284
# estimators = [estimators_list[0],estimators_list[5],
#               estimators_list[6],estimators_list[4]]

print(estimators)
stack = StackingRegressor(estimators=estimators,final_estimator = RidgeCV())
# stack = StackingRegressor(estimators=estimators,final_estimator = LassoCV())


# show CV performance of the selected stacking model
y_pred, scores= show_CV_performance(X, y, stack, nfold=nfold, title='stack')
scores


#%% plot single model vs stacking
estimators_all = estimators + [('Stacking model', stack)]

# plot all in one fig
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
axs = np.ravel(axs)

for ax, (name, est) in zip(axs, estimators_all):
    start_time = time.time()
    score = cross_validate(est, X, y,
                           scoring=['r2', 'neg_mean_squared_error'],
                           cv=nfold,
                           n_jobs=-1, verbose=1)
    elapsed_time = time.time() - start_time

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0, cv=nfold)

    plot_regression_results(
        ax, y, y_pred,
        name,
        (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$RMSE={:.3f} \pm {:.2f}$')
        .format(np.mean(score['test_r2']),
                np.std(score['test_r2']),
                np.mean(np.sqrt(-score['test_neg_mean_squared_error'])),
                np.std(np.sqrt(-score['test_neg_mean_squared_error']))),
        elapsed_time)

plt.suptitle('Single model vs. stacking model')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


#%% predict using only stacking 
stack.fit(X, y) 
y_pred = stack.predict(X)
y_pred_final = stack.predict(final_test)

#%% save final test result
# sample=pd.read_csv('./data/sample_submission.csv')
# save_dir = os.path.join(os.getcwd(), 'submission')

# y_hat_list = [y_pred_final] 
# # y_hat_name = ['stack_all_laso']
# y_hat_name = ['stack_all_ridge']

# for y_hat, name in zip(y_hat_list, y_hat_name):
#     submission=pd.DataFrame({"Id":sample['Id'],
#                               "SalePrice": np.exp(y_hat)})
#     submission.to_csv(os.path.join(save_dir, name +'.csv'), index = False)









