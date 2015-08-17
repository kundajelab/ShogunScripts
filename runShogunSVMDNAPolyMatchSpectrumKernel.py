# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/serialization_string_kernels_modular.py

import numpy as np
from modshogun import StringCharFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, Labels
from modshogun import MSG_DEBUG
from modshogun import SVMLight
from modshogun import PolyMatchWordStringKernel
from modshogun import BinaryLabels, LibSVM
from modshogun import ROCEvaluation
import sys


TRAININGDATAFILENAME = sys.argv[1]
TRAININGLABELSFILENAME = sys.argv[2]
VALIDATIONDATAFILENAME = sys.argv[3]
VALIDATIONLABELSFILENAME = sys.argv[4]
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[5]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[6]
K = int(sys.argv[7])
SVMC = float(sys.argv[8]) # Initially 1
GAP = int(sys.argv[9]) # Initially 0


def makeStringList(stringFileName):
	# Get a string list from a file
	stringList = []
	stringFile = open(stringFileName)
	skippedLines = []
	
	lineCount = 0
	for line in stringFile:
		# Iterate through the string file and get the string from each line
		if "N" in line.strip():
			# The current sequence has an N, so skip it
			skippedLines.append(lineCount)
		else:
			stringList.append(line.strip())
		lineCount = lineCount + 1
	
	print len(skippedLines)
	stringFile.close()
	return [stringList, skippedLines]


def makeIntList(intFileName, skippedLines):
	# Get a float list from a file
	intList = []
	intFile = open(intFileName)
	
	lineCount = 0
	for line in intFile:
		# Iterate through the float file and get the float from each line
		if lineCount in skippedLines:
			# Skip the current line
			lineCount = lineCount + 1
			continue
		label = int(line.strip())
		if label == 0:
			# Labels are 1 and 0 instead of 1 and -1
			label = -1
		intList.append(label)
		lineCount = lineCount + 1
		
	intFile.close()
	return np.array(intList)
	
	
def runShogunSVMProteinPolyMatchSpectrumKernel(train_xt, train_lt, test_xt):
	"""
	run svm with spectrum kernel
	"""

    ##################################################
    # set up svm
	charfeat_train = StringCharFeatures(train_xt, DNA)
	feats_train = StringWordFeatures(DNA)
	feats_train.obtain_from_char(charfeat_train, K-1, K, GAP, False)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()
	
	charfeat_test = StringCharFeatures(test_xt, DNA)
	feats_test=StringWordFeatures(DNA)
	feats_test.obtain_from_char(charfeat_test, K-1, K, GAP, False)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()
	
	kernel=PolyMatchWordStringKernel(feats_train, feats_train, 2, True)
	kernel.io.set_loglevel(MSG_DEBUG)

    # init kernel
	labels = BinaryLabels(train_lt)
	
	# run svm model
	print "Ready to train!"
	svm=LibSVM(SVMC, kernel, labels)
	svm.io.set_loglevel(MSG_DEBUG)
	svm.train()

	# predictions
	print "Making predictions!"
	out1DecisionValues = svm.apply(feats_train)
	out1=out1DecisionValues.get_labels()
	kernel.init(feats_train, feats_test)
	out2DecisionValues = svm.apply(feats_test)
	out2=out2DecisionValues.get_labels()

	return out1,out2,out1DecisionValues,out2DecisionValues


def writeFloatList(floatList, floatListFileName):
	# Write a list of floats to a file
	floatListFile = open(floatListFileName, 'w+')
	
	for f in floatList:
		# Iterate through the floats and record each of them
		floatListFile.write(str(f) + "\n")
	floatListFile.close()


def outputResultsClassification(out1, out2, out1DecisionValues, out2DecisionValues, train_lt, test_lt):
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
	
	trainLabels = BinaryLabels(train_lt)
	evaluatorTrain = ROCEvaluation()
	evaluatorTrain.evaluate(out1DecisionValues, trainLabels)
	print "Training AUC:"
	print evaluatorTrain.get_auROC()
	
	numValidCorrect = 0
	numPosCorrect = 0
	numNegCorrect = 0
	for i in range(len(test_lt)):
		# Iterate through validation labels and count the number that are the same as the predicted labels
		if out2[i] == test_lt[i]:
			# The current prediction is correct
			numValidCorrect = numValidCorrect + 1
			if (out2[i] == 1) and (test_lt[i] == 1):
				# The prediction is a positive example
				numPosCorrect = numPosCorrect + 1
			else:
				numNegCorrect = numNegCorrect + 1
	fracValidCorrect = float(numValidCorrect)/float(len(test_lt))
	print "Validation accuracy:"
	print fracValidCorrect
	print "Number of correct positive examples:"
	print numPosCorrect
	print "Number of correct negative examples:"
	print numNegCorrect
	
	validLabels = BinaryLabels(test_lt)
	evaluatorValid = ROCEvaluation()
	evaluatorValid.evaluate(out2DecisionValues, validLabels)
	print "Validation AUC:"
	print evaluatorValid.get_auROC()


if __name__=='__main__':
	print('LibSVR')
	[train_xt, skippedLinesTrain] = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeIntList(TRAININGLABELSFILENAME, skippedLinesTrain)
	[test_xt, skippedLinesValid] = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeIntList(VALIDATIONLABELSFILENAME, skippedLinesValid)
	[out1, out2, out1DecisionValues, out2DecisionValues] = runShogunSVMProteinPolyMatchSpectrumKernel(train_xt, train_lt, test_xt)
	outputResultsClassification(out1, out2, out1DecisionValues, out2DecisionValues, train_lt, test_lt)