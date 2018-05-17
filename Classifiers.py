'''
Created on Mar 23, 2018

@author: Matthew Burke
'''


import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics.classification import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



#import the data as a DataFrame
df = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data", sep=',')

df = df.sample(frac=1).reset_index(drop=True) #shuffles the rows of the data randomly

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
#scale the input data
df[df.columns] = scaler.fit_transform(df[df.columns])

#split data into training and testing data
train = df.iloc[:500,:]
test = df.iloc[500:,:]

#separate data and classification labels
x_train = train.drop('whether he/she donated blood in March 2007', axis=1)
y_train = train.iloc[:,4] #classification of training data
x_test = test.drop('whether he/she donated blood in March 2007', axis=1)
y_test = test.iloc[:,4] #classification of testing data

'''
print("TRAIN")
print(train)
print(x_train)
print(y_train)
print("TEST")
print(test)
print(x_test)
print(y_test)
'''


def eval(y_test, y_model_pred_test, classifier): 
    print('Model parameters: ', classifier.get_params()) 
    acc = accuracy_score(y_test, y_model_pred_test, normalize=True, sample_weight=None)
    print('Overall Accuracy = ', acc)
    print(confusion_matrix(y_test, y_model_pred_test))
    print()
    print(classification_report(y_test, y_model_pred_test, target_names=['Did Not Donate', 'Donated']))
 

def tune(classifier, tuned_parameters):
    scores = ['accuracy', 'precision_macro', 'recall_macro']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(classifier, tuned_parameters, cv=5,
                           scoring='%s' % score)
        clf.fit(x_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        print("For model with best %s" % score)
        print("Overall Accuracy: ", acc)
        print()
        print(classification_report(y_true, y_pred))
        print()
   
def random_forest():
    forest = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                             bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                             warm_start=False, class_weight=None) 
    forest.fit(x_train, y_train)
    #y_forest_pred_train = forest.predict(x_train)
    y_forest_pred_test = forest.predict(x_test)
    print('RANDOM FOREST')
    eval(y_test, y_forest_pred_test, forest)
    
    # Set the parameters by cross-validation  'oob_score': (True, False),
    forest_params = [{'n_estimators': [30, 35, 45, 50, 55, 100], 'criterion': ('gini', 'entropy'), 'oob_score': (True, False), 'class_weight': ('balanced', 'balanced_subsample', None)}]
    tune(forest, forest_params)

def ada_boost():
    dtstump = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)
    
    #AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
    ada_boost = ensemble.AdaBoostClassifier(base_estimator=dtstump, n_estimators=55, learning_rate=1.0, algorithm='SAMME', random_state=None)
    ada_boost.fit(x_train, y_train)
    #y_ada_pred_train = ada_boost.predict(x_train)
    y_ada_pred_test = ada_boost.predict(x_test)
    print('ADA BOOST')
    eval(y_test, y_ada_pred_test, ada_boost)
    
    ada_params = [{'n_estimators': [35, 45, 50, 55, 100], 'learning_rate': [1.0, 2.0], 'algorithm':('SAMME','SAMME.R')}]
    tune(ada_boost, ada_params)


'''
***************** TASK 2 *************************
'''

def neural_net():
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    skl_nn.fit(x_train, y_train)
    y_nn_pred_train = skl_nn.predict(x_train)
    #y_nn_pred_train = np.argmax(y_nn_pred_train, axis=1)
    y_nn_pred_test = skl_nn.predict(x_test)
    #y_nn_pred_test = np.argmax(y_nn_pred_test, axis=1)
    print("NEURAL NETWORK MLPC")
    eval(y_test, y_nn_pred_test, skl_nn)
    
    nn_params = [{'hidden_layer_sizes':[(50, ), (100, ), (200, )], 'activation':('identity', 'logistic', 'tanh', 'relu'), 
                  'solver': ('lbfgs', 'sgd','adam'), 'learning_rate_init':[0.001, 0.01], 'tol':[0.0001, 0.0002], 'momentum':[0.9, 0.7]}]
    tune(skl_nn, nn_params)



def k_nearest_neighbors():
    #uses the best parameters found using tune()
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    knn.fit(x_train, y_train)
    #y_knn_pred_train = knn.predict(x_train)
    y_knn_pred_test = knn.predict(x_test)
    print('K NEAREST NEIGHBOR')
    eval(y_test, y_knn_pred_test, knn)
    
    #algorithm, p, metric, metric_params, and n_jobs were left at their default values since these values are most reasonable
    knn_params = {'n_neighbors': [1, 5, 7, 10, 15, 20], 'weights': ('uniform', 'distance'),'leaf_size':[25, 30, 35]}
    tune(knn, knn_params)

def logistic_reg():
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    log_reg.fit(x_train, y_train)
    y_log_reg_pred_train = log_reg.predict(x_train)
    y_log_reg_pred_test = log_reg.predict(x_test)
    print("LOGISTIC REGRESSION")
    eval(y_test, y_log_reg_pred_test, log_reg)
    
    log_reg_params = [{'tol':[0.0001, 0.0002], 'C':[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], 'fit_intercept':(True, False), 'class_weight':(None, 'balanced'), 'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}]
    tune(log_reg, log_reg_params)

def naive_bayes():
    nb = GaussianNB(priors=None)
    nb.fit(x_train, y_train)
    #y_nb_pred_train = nb.predict(x_train)
    y_nb_pred_test = nb.predict(x_test)
    print("NIAVE BAYES")
    eval(y_test, y_nb_pred_test, nb)
    #no hyper-parameters to tune with this model
    
def decision_tree():
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)
    dtree.fit(x_train, y_train)
    #y_dtree_pred_train = dtree.predict(x_train)
    y_dtree_pred_test = dtree.predict(x_test)
    print("DECISION TREE")
    eval(y_test, y_dtree_pred_test, dtree)
    
    dtree_params = [{'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'class_weight':(None, 'balanced')}]
    tune(dtree, dtree_params)

def majority_vote_5_models():
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    nb = GaussianNB(priors=None)
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False) 
    
    vote_5_uw = VotingClassifier(estimators=[('neural net', skl_nn), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='hard')
    vote_5_uw.fit(x_train, y_train) 

    #y_v5uw_pred_train = vote_5_uw.predict(x_train)
    y_v5uw_pred_test = vote_5_uw.predict(x_test)
    print("VOTE 5 UNWEIGHTED")
    eval(y_test, y_v5uw_pred_test, vote_5_uw)

def weighted_vote_5_models():  
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    nb = GaussianNB(priors=None)
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False) 
    
    vote_5_w = VotingClassifier(estimators=[('neural net', skl_nn), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='soft', weights=None)
    vote_5_w.fit(x_train, y_train)
    #y_v5w_pred_train = vote_5_w.predict(x_train)
    y_v5w_pred_test = vote_5_w.predict(x_test)
    print("VOTE 5 WEIGHTED")
    eval(y_test, y_v5w_pred_test, vote_5_w)
    vote_5_w_params = [{'weights':(None, [.79,.79,.75,.76,.71]), 'voting':('soft', 'hard')}]
    tune(vote_5_w, vote_5_w_params)


'''
******* TASK 3 ***************************************
'''
def majortiy_vote_7_models():
    #for use with ada boost
    dtstump = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)
     
    forest = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                             bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                             warm_start=False, class_weight=None) 
    ada_boost = ensemble.AdaBoostClassifier(base_estimator=dtstump, n_estimators=55, learning_rate=1.0, algorithm='SAMME', random_state=None)
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    nb = GaussianNB(priors=None)
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)  
    
    vote_7_uw = VotingClassifier(estimators=[('neural net', skl_nn), ('forest', forest), ('ada boost', ada_boost), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='hard')
    vote_7_uw.fit(x_train, y_train)
    y_v7uw_pred_test = vote_7_uw.predict(x_test)
    print("VOTE 7 UNWEIGHTED")
    eval(y_test, y_v7uw_pred_test, vote_7_uw)
      
def weighted_vote_7_models(): 
    #for use with ada boost
    dtstump = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)
    
    forest = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                             bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                             warm_start=False, class_weight=None) 
    ada_boost = ensemble.AdaBoostClassifier(base_estimator=dtstump, n_estimators=55, learning_rate=1.0, algorithm='SAMME', random_state=None)
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    nb = GaussianNB(priors=None)
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False) 
    
    vote_7_w = VotingClassifier(estimators=[('neural net', skl_nn), ('forest', forest), ('ada boost', ada_boost), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='soft', weights=[.73,.80,.79,.79,.75,.76,.71])
    vote_7_w.fit(x_train, y_train)
    y_v7w_pred_test = vote_7_w.predict(x_test)
    print("VOTE 7 WEIGHTED")
    eval(y_test, y_v7w_pred_test, vote_7_w)
    vote_7_w_params = [{'weights':(None, [.73,.80,.79,.79,.75,.76,.71]), 'voting':('soft', 'hard')}]
    tune(vote_7_w, vote_7_w_params)

def all_models_accuracy():
    
    #for use with ada boost
    dtstump = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False)
    
    forest = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                             bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                             warm_start=False, class_weight=None) 
    ada_boost = ensemble.AdaBoostClassifier(base_estimator=dtstump, n_estimators=55, learning_rate=1.0, algorithm='SAMME', random_state=None)
    skl_nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0002, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=25, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=2.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    nb = GaussianNB(priors=None)
    dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                   random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, class_weight=None, presort=False) 
    vote_5_uw = VotingClassifier(estimators=[('neural net', skl_nn), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='hard')
    vote_5_w = VotingClassifier(estimators=[('neural net', skl_nn), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='soft', weights=None)
    vote_7_uw = VotingClassifier(estimators=[('neural net', skl_nn), ('forest', forest), ('ada boost', ada_boost), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='hard')
    vote_7_w = VotingClassifier(estimators=[('neural net', skl_nn), ('forest', forest), ('ada boost', ada_boost), ('knn', knn), ('log reg', log_reg), ('nb', nb), ('dtree', dtree)], voting='soft', weights=None)
    
    for clf, label in zip([skl_nn, forest, ada_boost, knn, log_reg, nb, dtree, vote_5_uw, vote_5_w, vote_7_uw, vote_7_w], ['Neural Net', 'Random Forest', 'Ada Boost', 'K Nearest Neighbor', 'Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Ensemble 5 Unweighted', 'Ensemble 5 Weighted', 'Ensemble 7 Unweighted', 'Ensemble 7 Weighted']):
        scores = cross_val_score(clf, x_test, y_test, cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#uncomment each function to see its accuracy and a grid search of its parameters
print("PART 1")
#random_forest()
#ada_boost()
print("PART 2")
#neural_net()
#k_nearest_neighbors()
#logistic_reg()
#naive_bayes()
#decision_tree()
#majority_vote_5_models()
#weighted_vote_5_models()
print("PART 3")
#majortiy_vote_7_models()
#weighted_vote_7_models()   
all_models_accuracy() 


