from sklearn import svm
import sys
import kernels
import dataSplit
import pickle
import numpy as np
import src.kernels_c as kernels_c
import time
from wk_ngk_Kernels import wk, ngk
from sklearn.metrics import average_precision_score, f1_score, accuracy_score


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
def evaluate_model_new(test_labels, predictions, topic):
    pass

def evaluate_model(test_labels, predictions, topic):
    tp = 0
    fp=0
    tn=0
    fn=0
    F1=0
    precision=0
    
    for i in range(len(test_labels)):
        if predictions[i] == 1:
            if test_labels[i] == 1:
                tp=tp+1
            else:
                fp=fp+1
        else:
            if test_labels[i] == 0:
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
    print(test_labels)

    train_labels_bool=(np.array(train_labels)==topic)
    test_labels_bool=(np.array(test_labels)==topic)
    train_labels_bool = train_labels_bool*1
    test_labels_bool = test_labels_bool*1
    #print(train_labels_bool)
    
    print('Generating Training Gram matrix...')
    gram_matrix_train=kernels_c.get_gram_matrix(k_func, train_datapoints)
    print(gram_matrix_train )
    print("trained gram")
    timestr = time.strftime("%m%d%H%M")
    dataSplit.saving_data(gram_matrix_train, '../data/kernels/ngk_gram_train'+'_k'+str(k)+'_l'+str(lambdaDecay*10))
    print('Gram matrix generated...')
    
    classifier_training = svm.SVC(kernel ='precomputed')
    classifier = classifier_training.fit(gram_matrix_train, train_labels_bool)
    
    
    print('Generating Training Test matrix...')

    gram_matrix_test = kernels_c.get_gram_matrix(k_func, train_datapoints, test_datapoints)
    print(gram_matrix_test.T)
    print("test gram")
    timestr = time.strftime("%m%d%H%M")
    dataSplit.saving_data(gram_matrix_test, '../data/kernels/ngk_gram_test'+'_k'+str(k)+'_l'+str(lambdaDecay*10))
    test_labels_pred = classifier.predict(gram_matrix_test.T)
    
    print(test_labels_pred)

    #return 1
    return evaluate_model(test_labels_bool, test_labels_pred, topic)
def just_train_and_test(gram_matrix_train, gram_matrix_test, traindata, testdata, topic):
    
    train_datapoints,train_labels=zip(*traindata)
    test_datapoints,test_labels=zip(*testdata)
    

    train_labels_bool=(np.array(train_labels)==topic)
    test_labels_bool=(np.array(test_labels)==topic)
    train_labels_bool = train_labels_bool*1
    test_labels_bool = test_labels_bool*1

    #print(train_labels_bool)
    
    classifier_training = svm.SVC(kernel ='precomputed')
    #print(test_labels_bool)
    classifier = classifier_training.fit(gram_matrix_train, train_labels_bool)
    test_labels_pred = classifier.predict(gram_matrix_test)
    
    #print(len(gram_matrix_test))
    #print(gram_matrix_test)
    #print(len(gram_matrix_test[1]))
    #print(len(gram_matrix_test[4]))

    #print(test_labels_pred)

    return evaluate_model(test_labels_bool, test_labels_pred, topic)
    a = (test_labels_bool, test_labels_pred)
    #print(a)
    y_score = classifier.decision_function(gram_matrix_test)
    #print(y_score)
    #return 0
    #print(test_labels_bool, a)
    x = [average_f1_score(test_labels_bool, y_score, average=None), average_precision_score(test_labels_bool, y_score), recall_score(test_labels_bool, y_score, average=None)]
    return x

if __name__=='__main__':
    #trainGram = dataSplit.load_data('../data/kernels/ssk_gram_train01211902_k5_l5.0')
    #testGram = dataSplit.load_data('../data/kernels/ssk_gram_test01211905_k5_l5.0')


    traindata = dataSplit.load_data('../data/datasets/train_short_new')
    testdata = dataSplit.load_data('../data/datasets/test_short_new')
    #traindata = dataSplit.load_data('../data/datasets/train_only2')
    #testdata = dataSplit.load_data('../data/datasets/test_only2')
    #sampledata = dataSplit.load_data('../data/datasets/test_data_small')
    #print(sampledata[1])

    #VARIABLES for SSK
    k=7
    lambdaDecay=0.5
    topic='corn'


    #print(trainGram)
    #print(testGram)

    # result = just_train_and_test(trainGram, testGram.T, traindata, testdata, topic)
    # print(result)
    #print(wk(traindata[0][0],traindata[1][0]))
    for k in [7,11]:
        for lambdaDecay in [0.5]:
            result = run_experiment(kernels_c.ssk(k,lambdaDecay), traindata, testdata, topic, k, lambdaDecay)
            print(result)

    #result = run_experiment(ngk, traindata, testdata, topic, k, lambdaDecay)
    print(result)






