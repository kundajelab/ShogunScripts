# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/serialization_string_kernels_modular.py

import numpy as np
from modshogun import StringCharFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, Labels
from modshogun import MSG_DEBUG
from modshogun import SVMLight
from modshogun import CommWordStringKernel
from modshogun import BinaryLabels, LibSVMOneClass,RealFeatures
from modshogun import ROCEvaluation

import argparse
import sys


def parseArgument():
	parser = argparse.ArgumentParser(description='Run Shogun One Class SVM DNA Spectrum Kernel')
	parser.add_argument('string', metavar='[train data]', nargs=1, help='Training Data File, format: sequence (ATCG) one instance per line' )
	parser.add_argument('string', metavar='[train labels]',  nargs=1, help='Training Labels File, format: 0/1 where each label of an instance matches the data line number (because this is One-class SVM training set should be all one class therefore should be all "1"' )
	parser.add_argument('string', metavar='[validation data]',  nargs=1, help='Validation Data File, same format as training data' )
	parser.add_argument('string', metavar='[validation label]',  nargs=1, help='Validation Lables File, same format as training label (can be either 0 or 1)' )
	parser.add_argument('string', metavar='[train classification]',  nargs=1, help='Classification result on training set' )
	parser.add_argument('string', metavar='[validation classification]',  nargs=1, help='Classification result on validation set' )
	parser.add_argument('int', metavar='[k]',  nargs=1, help='Value for K' )
	parser.add_argument('float', metavar='[SVM C]',  nargs=1, help='SVM C' )
	parser.add_argument('int', metavar='[number of Gaps]',  nargs=1, help='Number of Gaps' )
	parser.add_argument('float', metavar='[epsilon]',  nargs=1, help='Value of epsilon' )
	args = parser.parse_args()

	global TRAININGDATAFILENAME 
	global TRAININGLABELSFILENAME 
	global VALIDATIONDATAFILENAME 
	global VALIDATIONLABELSFILENAME 
	global TRAINPREDICTIONSEPSILONFILENAME
	global VALIDATIONPREDICTIONSEPSILONFILENAME 
	global SVMC 
	global GAP
	global K 
	global EPSILON
	
	TRAININGDATAFILENAME = sys.argv[1]
	TRAININGLABELSFILENAME = sys.argv[2]
	VALIDATIONDATAFILENAME = sys.argv[3]
	VALIDATIONLABELSFILENAME = sys.argv[4]
	TRAINPREDICTIONSEPSILONFILENAME = sys.argv[5]
	VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[6]
	K = int(sys.argv[7])
	SVMC = float(sys.argv[8]) # Initially 1
	GAP = int(sys.argv[9]) # Initially 0
	EPSILON = float(sys.argv[10])

def makeStringList(stringFileName):
	# Get a string list from a file
	stringList = []
	stringFile = open(stringFileName)
	skippedLines = []
	
	lineCount = 0
	for line in stringFile:
		# Iterate through the string file and get the string from each line
		if "N" in line.strip() or "n" in line.strip():
			# The current sequence has an N, so skip it
			skippedLines.append(lineCount)
		else:
			stringList.append(line.strip().upper())
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
	
	
def runShogunOneClassSVMDNASpectrumKernel(train_xt, train_lt, test_xt):
	"""
	run svm with spectrum kernel
	"""

    ##################################################
    # set up svr
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
	
	kernel=CommWordStringKernel(feats_train, feats_train, False)
	kernel.io.set_loglevel(MSG_DEBUG)

    # init kernel
	labels = BinaryLabels(train_lt)
	
	# run svm model
	print "Ready to train!"
	svm=LibSVMOneClass(SVMC, kernel)
	svm.set_epsilon(EPSILON)
	svm.train()


	# predictions
	print "Making predictions!"
	out1DecisionValues = svm.apply(feats_train)
	out1=out1DecisionValues.get_labels()
	kernel.init(feats_train, feats_test)
	out2DecisionValues = svm.apply(feats_test)
	out2=out2DecisionValues.get_labels()


#	predictions = svm.apply(feats_test)
#	return predictions, svm, predictions.get_labels()

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


def outputResultsClassificationWithMajorityClass(out1, out2, out1DecisionValues, out2DecisionValues, train_lt, test_lt, test_majorityClass):
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
	numMajorityClassCorrect = 0
	numMinorityClassCorrect = 0
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
			if test_majorityClass[i] == 1:
				numMajorityClassCorrect = numMajorityClassCorrect + 1
			else:
				numMinorityClassCorrect = numMinorityClassCorrect + 1
	fracValidCorrect = float(numValidCorrect)/float(len(test_lt))
	print "Validation accuracy:"
	print fracValidCorrect
	print "Fraction of correct positive examples:"
	print float(numPosCorrect)/float(len(np.where(test_lt > 0)[0]))
	print "Fraction of correct negative examples:"
	print float(numNegCorrect)/float(len(np.where(test_lt <= 0)[0]))
	print "Fraction of correct majority class examples:"
	print float(numMajorityClassCorrect)/float(len(np.where(test_majorityClass > 0)[0]))
	print "Fraction of correct minority class examples:"
	print float(numMinorityClassCorrect)/float(len(np.where(test_majorityClass <= 0)[0]))
	
	validLabels = BinaryLabels(test_lt)
	evaluatorValid = ROCEvaluation()
	evaluatorValid.evaluate(out2DecisionValues, validLabels)
	print "Validation AUC:"
	print evaluatorValid.get_auROC()


if __name__=='__main__':
	parseArgument() #print usage if not correctly used
	print('LibSVM')
	[train_xt, skippedLinesTrain] = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeIntList(TRAININGLABELSFILENAME, skippedLinesTrain)
	[test_xt, skippedLinesValid] = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeIntList(VALIDATIONLABELSFILENAME, skippedLinesValid)
	[out1, out2, out1DecisionValues, out2DecisionValues] = runShogunOneClassSVMDNASpectrumKernel(train_xt, train_lt, test_xt)
#	if len(sys.argv) > 10:
#		# There is majority class information
#		validationMajorityClassFileName = sys.argv[10]
#		test_majorityClass = makeIntList(validationMajorityClassFileName, skippedLinesValid)
#		outputResultsClassificationWithMajorityClass(out1, out2, out1DecisionValues, out2DecisionValues, train_lt, test_lt, test_majorityClass)
#	else:
	outputResultsClassification(out1, out2, out1DecisionValues, out2DecisionValues, train_lt, test_lt)
