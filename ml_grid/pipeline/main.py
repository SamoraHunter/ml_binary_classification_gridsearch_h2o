import traceback

import ml_grid
import numpy as np
# from ml_grid.model_classes.adaboost_classifier_class import adaboost_class
# from ml_grid.model_classes.gaussiannb_class import GaussianNB_class
# from ml_grid.model_classes.gradientboosting_classifier_class import \
#     GradientBoostingClassifier_class
# from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
# from ml_grid.model_classes.knn_classifier_class import knn_classifiers_class
# from ml_grid.model_classes.knn_gpu_classifier_class import \
#     knn__gpu_wrapper_class
# from ml_grid.model_classes.logistic_regression_class import \
#     LogisticRegression_class
# from ml_grid.model_classes.mlp_classifier_class import mlp_classifier_class
# from ml_grid.model_classes.quadratic_discriminant_class import \
#     quadratic_discriminant_analysis_class
# from ml_grid.model_classes.randomforest_classifier_class import \
#     RandomForestClassifier_class
# from ml_grid.model_classes.svc_class import SVC_class
# from ml_grid.model_classes.xgb_classifier_class import XGB_class_class
from ml_grid.model_classes.h2o_classifier_class import h2o_classifier_class
#from ml_grid.model_classes import LogisticRegression_class
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util import grid_param_space
from sklearn.model_selection import ParameterGrid

# +
class run():
    
    
    def __init__(self, ml_grid_object, local_param_dict): # kwargs**
        
        
        input_csv_path = '/home/aliencat/samora/HFE/HFE/v20/30163_to_16408_imputed_outcome_grid.csv'

        #instead get the original object...?
        #self.ml_grid_object = ml_grid.pipeline.data.pipe(input_csv_path, drop_term_list=['chrom', 'hfe'])
        
        self.ml_grid_object = ml_grid_object
        
        
        self.model_class_list = [
            
        h2o_classifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train)
            
#         LogisticRegression_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         knn_classifiers_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         quadratic_discriminant_analysis_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),

#         SVC_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         XGB_class_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
   
#         mlp_classifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         RandomForestClassifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         GradientBoostingClassifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         kerasClassifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         GaussianNB_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
#         adaboost_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
        
        #knn__gpu_wrapper_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size='medium'),
   
    ]
        
        
        for elem in self.model_class_list:
            
            pg = ParameterGrid(elem.parameter_space) #
            
            
            print(elem.method_name)
            print(len(pg))
            for param in elem.parameter_space:
                try:
                    if(type(param)is not list):
                        if(isinstance(elem.parameter_space.get(param), list) is False and isinstance(elem.parameter_space.get(param), np.ndarray) is False):
                            print(elem.method_name, param)
                            print(type(elem.parameter_space.get(param)))
                except Exception as e:
                    print(e)
                    pass

        scores_tuple_list = []
        model_error_list = []


        self.arg_list = []
        for model_class in self.model_class_list:

            class_name = model_class

            self.arg_list.append(
                (
                    class_name.algorithm_implementation,
                    class_name.parameter_space,
                    class_name.method_name,
                    self.ml_grid_object
                )
            )
            
            
        self.multiprocess = False
        
        self.local_param_dict = local_param_dict
        
        
        
    def execute(self):
        
        
        
        #print(self.arg_list)
        
        #print(type(self.arg_list[0][0]))
        
        self.model_error_list = []
        
        if self.multiprocess == True:

            def multi_run_wrapper(args):
                return grid_search_cross_validate(*args)

            if __name__ == "__main__":
                from multiprocessing import Pool

                pool = Pool(8)
                results = pool.map(multi_run_wrapper, self.arg_list)
                # print(results)
                pool.close() # exp


        if self.multiprocess == False:
            for k in range(0, len(self.arg_list)):
                try:
                    print("grid searching...")
                    grid_search_cross_validate.grid_search_crossvalidate(
                        *self.arg_list[k]
                        
                        #algorithm_implementation = LogisticRegression_class(parameter_space_size='medium').algorithm_implementation, parameter_space = self.arg_list[k][1], method_name=self.arg_list[k][2], X = self.arg_list[k][3], y=self.arg_list[k][4]
                    )
                except Exception as e:
                    
                    print(e)
                    print("error on ", self.arg_list[k][2])
                    self.model_error_list.append([self.arg_list[k][0], e, traceback.print_exc()] )  


        print("Model error list: nb. errors returned from func")
        print(self.model_error_list)
        
        return self.model_error_list
