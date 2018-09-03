import utils
import pandas as pd
import numpy as np
import etl
import models_partc

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *


RANDOM_STATE = 545510477

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
    X_train, Y_train = utils.get_data_from_svmlight('../deliverables/features_svmlight.train')
    
    deliverables_path = '../deliverables/'
    test_events = pd.read_csv('../data/test/events.csv')
    test_events_map = pd.read_csv('../data/test/event_feature_map.csv')
    
    test_aggregated_events = etl.aggregate_events(test_events, None, test_events_map, deliverables_path)
    
    #make patient_features for test data
    test_patient_features = test_aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda g: list(map(tuple, g.values.tolist()))).to_dict()
    
    #store test_feature.txt and test_svmlight file
    
    line_svm = ''
    line_test = ''

    for key in sorted(test_patient_features):
        line_svm +='1 '
        line_test += str(int(key)) +' '

        for tup in sorted(test_patient_features[key]):
            line_svm += str(int(tup[0])) + ':' + str("{:.6f}".format(tup[1])) + ' '
            line_test += str(int(tup[0])) + ':' + str("{:.6f}".format(tup[1])) + ' '
        line_svm += '\n' 
        line_test += '\n'
        
    test_featuresfile = open(deliverables_path + 'test_features.txt', 'wb')
    test_svmlightfile = open(deliverables_path + 'test_mymodel_svm.train','wb')   
    test_svmlightfile.write(bytes(line_svm,'UTF-8')) #Use 'UTF-8'
    test_featuresfile.write(bytes(line_test,'UTF-8'))  
    
    test_data = load_svmlight_file(deliverables_path + 'test_mymodel_svm.train',n_features=3190)
    X_test = test_data[0]
    
    return X_train, Y_train, X_test
    

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
    #logisticRegr = LogisticRegression(random_state = RANDOM_STATE, max_iter = 50)
    #logisticRegr.fit(X_train, Y_train)
 #   sample_leaf_options = [1,5,10,50,200,500]
  #  for leaf_size in sample_leaf_options:
    rf = RandomForestClassifier(random_state=RANDOM_STATE,oob_score=True,
                                n_jobs = -1, n_estimators = 600, max_depth =110)
                                
    rf.fit(X_train,Y_train)
       
#    y_pred1 = rf.predict(X_train)
#    print(roc_auc_score(Y_train,y_pred1))
    
    Y_pred = rf.predict(X_test)
    return Y_pred
      


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)

	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	