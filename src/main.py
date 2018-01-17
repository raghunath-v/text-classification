from sklearn import svm
import kernels


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
def evaluate_model(test_labels, predictions):
    pass


def run_experiment(kernel, traindata, testdata):
    
    train_datapoints,train_labels=zip(*traindata)
    
    classifier_training = svm.SVC(kernel ='precomputed')
    classifier = classifier_training.fit(train_datapoints, train_labels)
    
    
if __name__=="__main__":
    traindata = load_data('../data/datasets/train')
    testdata = load_data('../data/datasets/test')
    
    #VARIABLES for SSK
    k=3
    lambdaDecay=0.5
    
    run_experiment(kernels.ssk(k,lambdaDecay), traindata, testdata)
    
    

    
    
    
    

