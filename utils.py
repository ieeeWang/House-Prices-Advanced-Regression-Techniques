import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn import preprocessing
import time
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR

#%%
def SFS_stack_models(estimators_list, X, y, nfold=10):
    """
    Sequential Forward Selection (SFS) model stacking.
    SFS starts from the 1st entry of estimators_list.
    """
    # SFS initial state
    estimators_sel = [] #seleted set
    estimators_rem = estimators_list.copy() #remaining set
    
    # SFS starts from the 1st entry of estimators_list
    estimators_sel = [estimators_rem.pop(0)]
    
    RMSE_best = []; RMSE_all_steps = []; 
    while estimators_rem:
        print('number of estimators in remaining set:', len(estimators_rem))
        # compute score of each entry in the remaining set
        RMSE_mean = []; 
        for estimator in estimators_rem:
            estimators_sel.append(estimator) # add an estimator
            
            stack = StackingRegressor(estimators = estimators_sel,
                                       # final_estimator = LassoCV()
                                        final_estimator = RidgeCV()
                                      )
            _, scores = show_CV_performance(X, y, stack,nfold=nfold,plot=False)
            RMSE_mean.append(scores['RMSE_mean'])
            
            estimators_sel.pop() # remove the added estimator
           
            
        # find the estimator that help reduce RMSE the most
        loc = RMSE_mean.index(np.min(RMSE_mean))
        print('selected estimator:', estimators_rem[loc])
        print('min RMSE on current step:', np.min(RMSE_mean))
        RMSE_best.append(np.min(RMSE_mean)) 
        RMSE_all_steps.append(RMSE_mean)
        # move it from estimators_rem to estimators_sel
        estimators_sel.append(estimators_rem.pop(loc))
          
    return  estimators_sel, RMSE_best, RMSE_all_steps 



def plot_lasso_path(model, t_lasso_cv):
    """
    Plot regularization path of fitted lasso model using the coordinate descent.
    """
    plt.figure(figsize=(6, 3))
    # This is to avoid division by zero while doing np.log10
    EPSILON = 1e-6
    plt.semilogx(model.alphas_ + EPSILON, model.mse_path_, ':')
    plt.plot(model.alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(model.alpha_ + EPSILON, linestyle='--', color='k',
                label='alpha: CV estimate')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
              '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')



def show_CV_performance(X, y, model, nfold=5, plot = True, title='Test model'):
    """
    This function shows the n-fold CV performance given a model with its parameters.
    And return its n-fold perdiction, in which each entry is a prediction.
    """
    # show CV performance
    start_time = time.time()
    score = cross_validate(model, X, y,
                           scoring=['r2', 'neg_mean_squared_error'],
                           cv=nfold,
                           n_jobs=-1, verbose=1)
    elapsed_time = time.time() - start_time
    
    r2_mean = np.mean(score['test_r2'])
    r2_std =  np.std(score['test_r2'])         
    RMSE_mean = np.mean(np.sqrt(-score['test_neg_mean_squared_error']))
    RMSE_std = np.std(np.sqrt(-score['test_neg_mean_squared_error']))
                   
    y_pred = cross_val_predict(model, X, y, n_jobs=-1, verbose=0, cv=nfold)
    
    if plot:      
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        scores_text=(r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$RMSE={:.3f} \pm {:.2f}$'
                       ).format(r2_mean, r2_std, RMSE_mean, RMSE_std)
        plot_regression_results(ax, y, y_pred, title, scores_text, elapsed_time)

    scores = {}
    scores['r2_mean'] = r2_mean
    scores['r2_std'] = r2_std
    scores['RMSE_mean'] = RMSE_mean
    scores['RMSE_std'] = RMSE_std
    
    return y_pred, scores


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds on 10-fold CVs'.format(elapsed_time)
    ax.set_title(title)  
    
    
def Categorical2Numerical(df_train_cat, df_test_cat):
    """
    This function converts all Categorical columns to numerical for both 
    train and test sets. Each categorical variable will be replaced by its 
    corresponding mean value of log(saleprice).
    INPUT: [df_train_cat.columns] == [df_test_cat.columns, 'Saleprice_Log']
    """
    for catg in df_train_cat.columns[:-1]: # SalePrice_Log at last column
        g = df_train_cat.groupby(catg)['SalePrice_Log'].mean()
        # print(g) # Series is dict-like, e.g., g['Ex'] = 12.633614
        # update categorical variable
        df_train_cat[catg] = df_train_cat[catg].map(g)
        df_test_cat[catg] = df_test_cat[catg].map(g)


def standardization_train_test(df_train_num, df_test_num):
    """
    Standardization of datasets is a common requirement for many machine 
    learning estimators. It takes mean removal (to 0) and variance scaling to 1. 
    Note that both the train and test sets were standardized at the same time, 
    so taht the same linear transformation was performed.
    """    
    df_num_traintest = pd.concat((df_train_num, df_test_num), 
                                 sort=False).reset_index(drop=True)
    # print('train+test:', df_num_traintest.shape)
    X_scaled = preprocessing.scale(df_num_traintest) # each column: (m, sd) ~(0, 1)
    
    df_train_num2 = pd.DataFrame(X_scaled[:df_train_num.shape[0]])# first n rows
    df_test_num2 = pd.DataFrame(X_scaled[-df_test_num.shape[0]:])# last m rows
    
    df_train_num2.columns = df_num_traintest.columns
    df_test_num2.columns = df_num_traintest.columns
    # print('train after standardization:',df_train_num2.shape)
    # print('test after standardization:',df_test_num2.shape)
    
    return df_train_num2, df_test_num2


def plot_predicted(Y, Y_pred):
    sns.set(font_scale=1.5)
    plt.figure(figsize=(10,6))
    sns.scatterplot(Y, Y_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs. Predicted Prices")


def get_best_score(grid):    
    best_score = np.sqrt(-grid.best_score_)
    print('best RMSE:', best_score)    
    print('best params:',grid.best_params_)
    print('best model:', grid.best_estimator_)
    
    return best_score    