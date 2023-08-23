

import time
import traceback

import keras
import numpy as np
import pandas as pd
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from numpy import absolute, mean, std
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import (classification_report, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, RepeatedKFold,
                                     cross_validate)

random_grid_search = True

# +
sub_sample_param_space_pct = 0.00001

sub_sample_param_space_pct = 0.01
# -

grid_n_jobs = 4

metric_list = {'auc': make_scorer(roc_auc_score, needs_proba=False),
                'f1':'f1',
                'accuracy':'accuracy',
                'recall': 'recall'}

class grid_search_crossvalidate():
    
    
    def __init__(self, algorithm_implementation, parameter_space, method_name, ml_grid_object, **param_dict): # kwargs**
        
        self.global_parameters = global_parameters()
        
        self.ml_grid_object_iter = ml_grid_object
        
        self.X_train = self.ml_grid_object_iter.X_train
        
        self.y_train = self.ml_grid_object_iter.y_train
        
        self.X_test = self.ml_grid_object_iter.X_test
        
        self.y_test = self.ml_grid_object_iter.y_test
        
        self.X_test_orig = self.ml_grid_object_iter.X_test_orig
        
        self.y_test_orig = self.ml_grid_object_iter.y_test_orig
        
        
        
        
        #print("grid_search_crossvalidate called")
        
        self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # global scores_tuple_list
        # global i
        start = time.time()

        #print(method_name)

        current_algorithm = algorithm_implementation

        parameters = parameter_space
        n_iter_v = np.nan
    #     if(sub_sample_param_space):
    #         sub_sample_param_space_n = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))
    #         parameter_space random.sample(ParameterGrid(parameter_space), sub_sample_param_space_n)


        #Grid search over hyperparameter space, randomised. 
        if(random_grid_search):
            n_iter_v = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space))) + 2
            
            grid= RandomizedSearchCV(current_algorithm, parameters,
                                    verbose=1, cv=[(slice(None), slice(None))],
                                    n_jobs =grid_n_jobs, n_iter = n_iter_v, error_score=np.nan)
        else:   
            grid = GridSearchCV(
                current_algorithm, parameters, verbose=1, cv=[(slice(None), slice(None))], n_jobs = grid_n_jobs,
                error_score=np.nan
            )  # Negate CV in param search for speed

        pg = ParameterGrid(parameter_space)
        pg = len(pg)
        #print(pg)
        if pg > 100000:
            print("grid too large", str(pg))
            raise Exception("grid too large", str(pg))
        # print(grid)
        print("Full pg", pg)
        grid.fit(self.X_train, self.y_train)

        
        #Get cross validated scores for best hyperparameter model on x_train_/y_train
        if type(grid.estimator) is not keras.wrappers.scikit_learn.KerasClassifier:

            current_algorithm = grid.best_estimator_
            current_algorithm.fit(self.X_train, self.y_train)
            
        else:
            current_algorithm = KerasClassifier(
                build_fn=kerasClassifier_class.create_model(),
                verbose=0,
                layers=grid.best_params_["layers"],
                width=grid.best_params_["width"],
                learning_rate=grid.best_params_["learning_rate"],
            )
        
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
        

        if(self.global_parameters.debug_level == 3):
            
            
            debug_print_statements_class.debug_print_scores(scores)
        plot_auc = False
        if(plot_auc):    
            #This was passing a classifier trained on the test dataset....
            
            
            plot_auc_results(current_algorithm, self.X_test_orig[self.X_train.columns], self.y_test_orig, self.cv)
            #plot_auc_results(grid.best_estimator_, X_test_orig, self.y_test_orig, cv)
        
        #algorithm_implementation, parameter_space, method_name
        
        
        best_pred_orig = current_algorithm.predict(self.X_test_orig[self.X_test_orig.columns]) #exp
        
        print(f"type {type(best_pred_orig)}")
        print(f"shape {best_pred_orig.shape}")
        
        
        print(type(current_algorithm))
        print(current_algorithm)
        print(current_algorithm.get_leader_params())
        
        #best_pred_orig = best_pred_orig[0]
        #ml_grid_object, scores, best_pred_orig, current_algorithm, method_name, pg, start 
        
        #print("project_score_save_class.update_score_log..")
        project_score_save_class.update_score_log(
                                                        self = self,
                                                        ml_grid_object = self.ml_grid_object_iter,
                                                        scores = scores,
                                                        best_pred_orig = best_pred_orig,
                                                        current_algorithm = current_algorithm,
                                                        method_name = current_algorithm.get_leader_params(),
                                                        pg = pg,
                                                        start = start
                                                                 )
        
        
        
        
        
        
        


        #return (method_name, current_algorithm_scores)

        

     

        

   
