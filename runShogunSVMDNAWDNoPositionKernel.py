# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/kernel_comm_ulong_string_modular.py

import numpy as np
from numpy import arange, double
from modshogun import StringCharFeatures
from modshogun import DNA, Labels
from modshogun import MSG_DEBUG
from modshogun import SVMLight
from modshogun import BinaryLabels, LibSVM
from modshogun import WeightedDegreeStringKernel
import sys


TRAININGDATAFILENAME = sys.argv[1]
TRAININGLABELSFILENAME = sys.argv[2]
VALIDATIONDATAFILENAME = sys.argv[3]
VALIDATIONLABELSFILENAME = sys.argv[4]
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[5]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[6]
DEGREE = int(sys.argv[7])
SVMC = float(sys.argv[8]) # Initially 1


def makeStringList(stringFileName):
	# Get a string list from a file
	stringList = []
	stringFile = open(stringFileName)
	
	for line in stringFile:
		# Iterate through the string file and get the string from each line
		stringList.append(line.strip())
		
	stringFile.close()
	return stringList


def makeIntList(intFileName):
	# Get a float list from a file
	intList = []
	intFile = open(intFileName)
	
	for line in intFile:
		# Iterate through the float file and get the float from each line
		label = int(line.strip())
		if label == 0:
			# Labels are 1 and 0 instead of 1 and -1
			label = -1
		intList.append(label)
		
	intFile.close()
	return np.array(intList)
	
	
def runShogunSVMDNAWDNoPositionKernel(train_xt, train_lt, test_xt):
	"""
	run svm with non-position WD kernel
	"""

    ##################################################
    # set up svm
	feats_train = StringCharFeatures(train_xt, DNA)	
	feats_test = StringCharFeatures(test_xt, DNA)
	
	kernel=WeightedDegreeStringKernel(feats_train, feats_train, DEGREE)
	kernel.io.set_loglevel(MSG_DEBUG)
	
	weights=arange(1,DEGREE+1,dtype=double)[::-1]/ \
		sum(arange(1,DEGREE+1,dtype=double))
	kernel.set_wd_weights(weights)

    # init kernel
	labels = BinaryLabels(train_lt)
	
	# run svm model
	print "Ready to train!"
	svm=LibSVM(SVMC, kernel, labels)
	svm.io.set_loglevel(MSG_DEBUG)
	svm.train()

	# predictions
	print "Making predictions!"
	out1=svm.apply(feats_train).get_labels()
	kernel.init(feats_train, feats_test)
	out2=svm.apply(feats_test).get_labels()

	return out1,out2


def writeFloatList(floatList, floatListFileName):
	# Write a list of floats to a file
	floatListFile = open(floatListFileName, 'w+')
	
	for f in floatList:
		# Iterate through the floats and record each of them
		floatListFile.write(str(f) + "\n")
	floatListFile.close()


def outputResultsClassification(out1, out2, train_lt, test_lt):
	# Output the results to the appropriate output files
	writeFloatList(out1, TRAINPREDICTIONSEPSILONFILENAME)
	writeFloatList(out2, VALIDATIONPREDICTIONSEPSILONFILENAME)
	
	numTrainCorrect = 0
	for i in range(len(train_lt)):
		# Iterate through training labels and count the number that are the same as the predicted labels
		if out1[i] == train_lt[i]:
			# The current prediction is correct
			numTrainCorrect = numTrainCorrect + 1
	fracTrainCorrect = float(numTrainCorrect)/float(len(train_lt))
	print "Training accuracy:"
	print fracTrainCorrect
	
	numValidCorrect = 0
	for i in range(len(test_lt)):
		# Iterate through validation labels and count the number that are the same as the predicted labels
		if out2[i] == test_lt[i]:
			# The current prediction is correct
			numValidCorrect = numValidCorrect + 1
	fracValidCorrect = float(numValidCorrect)/float(len(test_lt))
	print "Validation accuracy:"
	print fracValidCorrect


if __name__=='__main__':
	print('LibSVR')
	train_xt = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeIntList(TRAININGLABELSFILENAME)
	test_xt = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeIntList(VALIDATIONLABELSFILENAME)
	[out1, out2] = runShogunSVMDNAWDNoPositionKernel(train_xt, train_lt, test_xt)
	outputResultsClassification(out1, out2, train_lt, test_lt)