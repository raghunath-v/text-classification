from sklearn import svm
import sys
import kernels
import dataSplit
import pickle
import numpy as np
import src.kernels_c as kernels_c
import time



#TODO
# What does the Gram Matrix?



# ----------------------------------------------------------
# Part 1 - Load the data och preprocessing the data into
# training dataset and test dataset
# ----------------------------------------------------------
# Load Dataset
def load_data(file_name):
    pass


# Split the data into training and test
def data_split(data):
    pass


# Clean the data
def preprocessing(dataset):
    pass

# ----------------------------------------------------------
# Part 2 - Using the different kernels and train the model
# ----------------------------------------------------------
# Feature extraction
def feature_extract(method, dataset):
    pass


# Train the model/models
def train_classifier(train_datapoints, train_labels):
    #classifier_training = SVC(kernel ='precomputed')
    #classifier = classifier_training.fit(train_datapoints, train_labels)
    #return classifier
    pass


# ----------------------------------------------------------
# Part 3 - Result and evaluate the model/result
# ----------------------------------------------------------
# Count scores
def score_model():
    pass


# Evaluate the models
def evaluate_model(test_labels, predictions, topic):
    tp = 0
    fp=0
    tn=0
    fn=0
    F1=0
    precision=0
    
    for i in range(len(test_labels)):
        if predictions[i] == topic:
            if test_labels[i] == topic:
                tp=tp+1
            else:
                fp=fp+1
        else:
            if test_labels[i] == topic:
                tn=tn+1
            else:
                fn=fn+1
                
    if tp+fp != 0:
        precision = float(tp)/float(tp+fp)
        
    recall = float(tp) / float(tp + fn)
    
    if precision+recall != 0:
        F1=2*precision*recall/(precision+recall)        
        
    return F1, precision, recall
    
        


def run_experiment(k_func, traindata, testdata, topic, k, lambdaDecay):
    
    train_datapoints,train_labels=zip(*traindata)
    test_datapoints,test_labels=zip(*testdata)
    
    print('Beginning training...')
    #print(np.array(train_labels))
    
    train_labels_bool=(np.array(train_labels)==topic)
    test_labels_bool=(np.array(test_labels)==topic)
    #print(train_labels_bool)
    
    
    print('Generating Training Gram matrix...')
    gram_matrix_train=kernels_c.get_gram_matrix(k_func, train_datapoints)
    timestr = time.strftime("%m%d%H%M")
    dataSplit.saving_data(gram_matrix_train, '../data/kernels/ssk_gram_train'+timestr +'_k'+str(k)+'_l'+str(lambdaDecay*10))
    print('Gram matrix generated...')
    
    classifier_training = svm.SVC(kernel ='precomputed')
    classifier = classifier_training.fit(gram_matrix_train, train_labels_bool)
    
    
    print('Generating Training Test matrix...')
    gram_matrix_test = kernels_c.get_gram_matrix(k_func, train_datapoints, test_datapoints)
    timestr = time.strftime("%m%d%H%M")
    dataSplit.saving_data(gram_matrix_test, '../data/kernels/ssk_gram_test'+timestr+'_k'+str(k)+'_l'+str(lambdaDecay*10))
    test_labels_pred = classifier.predict(gram_matrix_test)
    #return 1
    return evaluate_model(test_labels_bool, test_labels_pred, topic)

def just_train_and_test(gram_matrix_train, gram_matrix_test, traindata, testdata, topic):
    
    train_datapoints,train_labels=zip(*traindata)
    test_datapoints,test_labels=zip(*testdata)
    
    train_labels_bool=(np.array(train_labels)==topic)
    
    classifier_training = svm.SVC(kernel ='precomputed')
    classifier = classifier_training.fit(gram_matrix_train, train_labels_bool)
    test_labels_pred = classifier.predict(gram_matrix_test)
    
    return evaluate_model(test_labels, test_labels_pred, topic)
    
    
    
    

traindata = dataSplit.load_data('../data/datasets/train')
testdata = dataSplit.load_data('../data/datasets/test')
#sampledata = dataSplit.load_data('../data/datasets/test_data_small')
#print(sampledata[1])

#VARIABLES for SSK
k=5
lambdaDecay=0.5
topic='acq'
run_experiment(kernels_c.ssk(k,lambdaDecay), traindata, testdata, topic, k, lambdaDecay)
    
    

    
    
    
    

