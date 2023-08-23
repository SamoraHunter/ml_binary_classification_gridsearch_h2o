

import time
import traceback

import keras
import numpy as np
import pandas as pd
#from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save_h2o import project_score_save_class
from numpy import absolute, mean, std
#from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import (classification_report, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, RepeatedKFold,
                                     cross_validate)


from ml_grid.util.global_params import global_parameters

import warnings

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
#import tensorflow as tf

import h2o
from h2o.automl import H2OAutoML

from h2o.sklearn import H2OAutoMLClassifier

# +
#from ml_grid.model_classes.H2OAutoMLClassifier_wrapper import H2OAutoMLClassifier
# -

from ml_grid.model_classes.H2OAutoMLClassifier_wrapper import *


class h2o_evaluate_class():
    
    
    def __init__(self, algorithm_implementation, parameter_space, method_name, ml_grid_object, sub_sample_parameter_val = 100): # kwargs**
        #
        
        warnings.filterwarnings('ignore') 

        warnings.filterwarnings('ignore', category=FutureWarning)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        
        warnings.filterwarnings('ignore', category=UserWarning)
        
        h2o.init()
        
        self.global_params = global_parameters()
        
        self.verbose = self.global_params.verbose
        
        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct
        
        random_grid_search = self.global_params.random_grid_search
        
        self.sub_sample_parameter_val = sub_sample_parameter_val
        
        grid_n_jobs = self.global_params.grid_n_jobs
        
        
        self.metric_list = self.global_params.metric_list
        
        self.error_raise = self.global_params.error_raise
        
        
        if(self.verbose >=3):
            print(f"crossvalidating {method_name}")
        
        self.global_parameters = global_parameters()
        
        self.ml_grid_object_iter = ml_grid_object
        
        self.X_train = h2o.H2OFrame(self.ml_grid_object_iter.X_train)
        self.y_train = h2o.H2OFrame(self.ml_grid_object_iter.y_train.to_frame())
        self.X_test = h2o.H2OFrame(self.ml_grid_object_iter.X_test)
        self.y_test = h2o.H2OFrame(self.ml_grid_object_iter.y_test.to_frame())
        self.X_test_orig = h2o.H2OFrame(self.ml_grid_object_iter.X_test_orig)
        self.y_test_orig = h2o.H2OFrame(self.ml_grid_object_iter.y_test_orig.to_frame())

        
        self.nfolds_val = -1 #auto
        
        
        # Assign values to variables
        X_train_data = ml_grid_object.X_train
        y_train_data = ml_grid_object.y_train.to_frame()
        X_test_data = ml_grid_object.X_test
        y_test_data = ml_grid_object.y_test.to_frame()
        X_test_orig_data = ml_grid_object.X_test_orig
        y_test_orig_data = ml_grid_object.y_test_orig.to_frame()

        # Concatenate train and test data using Pandas
        train = pd.concat([X_train_data, y_train_data], axis=1)
        test = pd.concat([X_test_data, y_test_data], axis=1)
        test_orig = pd.concat([X_test_orig_data, y_test_orig_data], axis=1)

        train_h2o_frame = h2o.H2OFrame(train)
        test_h2o_frame = h2o.H2OFrame(test)


        outcome_variable = 'outcome_var_1'
        
        outcome_variable = y_train_data.columns[0]
        # Identify predictors and response
        x = list(train.columns)
        y = outcome_variable
        x.remove(y)

        # For binary classification, response should be a factor
        train_h2o_frame[y] = train_h2o_frame[y].asfactor()#.asnumeric()
        test_h2o_frame[y] = test_h2o_frame[y].asfactor()#.asnumeric()
        
        
        #self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
 
        start = time.time()

    
        aml = H2OAutoML(max_models=5, seed=1)
        aml.train(x=x, y=y, training_frame=train_h2o_frame)
        
        
        h2o.automl.get_leaderboard(aml, extra_columns = "ALL")

        if(self.global_parameters.verbose >= 4):
            
            debug_print_statements_class.debug_print_scores(scores)
            
        plot_auc = False
        if(plot_auc):    

            print(" ")
            
            
        best_model = aml.leader
            
        
        
        performance = best_model.model_performance(test_h2o_frame)
        
        
    
        #--------------------

        
        self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # global scores_tuple_list
        # global i
        start = time.time()

        current_algorithm = algorithm_implementation

        parameters = parameter_space
        n_iter_v = np.nan

        sklearn_model = H2OAutoMLClassifier(aml.leader)
        
        current_algorithm = sklearn_model
        
        
        current_algorithm.fit(self.X_train, self.y_train)
            

        scores = cross_validate(
            current_algorithm,
            self.X_train,
            self.y_train,
            scoring=metric_list,
            cv=self.cv,
            n_jobs=grid_n_jobs,  # Full CV on final best model #exp -1 was 1
            pre_dispatch = 80, #exp,
            error_score=np.nan
        )
        current_algorithm_scores = scores
    #     scores_tuple_list.append((method_name, current_algorithm_scores, grid))
        

        #best_pred_orig = current_algorithm.predict(self.X_test_orig[self.X_test_orig.columns]) #exp

        #--------------------
        
        predictions = best_model.predict(test_h2o_frame)
        
        best_pred_orig = predictions['predict'].as_data_frame().values
        
        
        pg = np.nan
        

#         this should be x_test...?
        #best_pred_orig = current_algorithm.predict(self.X_test[self.X_test.columns]) #exp
        
        project_score_save_class_h2o.update_score_log(
                                                        self = self,
                                                        ml_grid_object = self.ml_grid_object_iter,
                                                        scores = current_algorithm_scores,
                                                        best_pred_orig = best_pred_orig,
                                                        current_algorithm = current_algorithm,
                                                        method_name = method_name,
                                                        pg = pg,
                                                        start = start,
                                                        n_iter_v = n_iter_v
                                                                 )
        
        h2o.shutdown()

        

#         when to use validation set... and how to store which cases are in this valid set? can withold valid set even earlier...? should?

     

        

# +
#.asnumeric()   
