import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Extracts multiple sets of k-mers as features (k=(2,10)), and 
# trains a linear SVM on the data (only SVM practical given huge dataset size)

vect = CountVectorizer(analyzer="char",ngram_range=(2,10));
trainData = np.loadtxt("../data/SP1C_SELEX_TrainSeq",dtype='str');
trainlabels = np.loadtxt("../data/SP1C_SELEX_TrainLabels");
testData = np.loadtxt("../data/SP1C_SELEX_TestSeq",dtype='str'); 
testlabels = np.loadtxt("../data/SP1C_SELEX_TestLabels");

extractor = vect.fit(trainData);
trainData = extractor.transform(trainData);
testData = extractor.transform(testData);

clf = SGDClassifier(loss="hinge", alpha=0.8, penalty="l2")
svm = clf.fit(trainData, trainlabels);

trainPreds = svm.predict(trainData);
testPreds = svm.predict(testData);

print classification_report(trainlabels, trainPreds);
print accuracy_score(trainlabels, trainPreds);
print classification_report(testlabels, testPreds);
print accuracy_score(testlabels, testPreds);


