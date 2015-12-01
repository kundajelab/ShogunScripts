# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/serialization_string_kernels_modular.py

import numpy as np
from modshogun import StringCharFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, Labels
from modshogun import MSG_DEBUG
from modshogun import SVMLight
from modshogun import CombinedKernel
from modshogun import CombinedFeatures
from modshogun import WeightedCommWordStringKernel, WeightedDegreePositionStringKernel, CommWordStringKernel
from modshogun import LocalAlignmentStringKernel, FixedDegreeStringKernel, WeightedDegreeStringKernel
from modshogun import WeightedDegreeStringKernel, RegulatoryModulesStringKernel
from modshogun import MKLClassification
from modshogun import BinaryLabels, LibSVM
from modshogun import PluginEstimate
import sys


TRAININGDATAFILENAME = sys.argv[1]
TRAININGLABELSFILENAME = sys.argv[2]
VALIDATIONDATAFILENAME = sys.argv[3]
VALIDATIONLABELSFILENAME = sys.argv[4]
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[5]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[6]
K1 = int(sys.argv[7]) # starting K for multiple kernels
K2 = int(sys.argv[8]) # ending K for multiple kernels
SVMC = float(sys.argv[9]) # MKL C parameter
GAP = int(sys.argv[10]) # Initially 0

SHIFT = 35; # shift for weighted degree kernel

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
	
	
def runShogunSVMMultipleKernels(train_xt, train_lt, test_xt):
	"""
	Run SVM with Multiple Kernels
	"""

    ##################################################

    	# Take all examples
   	idxs = np.random.randint(1,14000,14000);
	train_xt = np.array(train_xt)[idxs];
    	train_lt = np.array(train_lt)[idxs];

    	# Initialize kernel and features
    	kernel=CombinedKernel()
	feats_train=CombinedFeatures()
	feats_test=CombinedFeatures()
	labels = BinaryLabels(train_lt)
	
	##################### Multiple Spectrum Kernels #########################
	for i in range(K1,K2,-1):
                # append training data to combined feature object
                charfeat_train = StringCharFeatures(list(train_xt), DNA)
                feats_train_k1 = StringWordFeatures(DNA)
                feats_train_k1.obtain_from_char(charfeat_train, i-1, i, GAP, False)
                preproc=SortWordString()
                preproc.init(feats_train_k1)
                feats_train_k1.add_preprocessor(preproc)
                feats_train_k1.apply_preprocessor()
                # append testing data to combined feature object
                charfeat_test = StringCharFeatures(test_xt, DNA)
                feats_test_k1=StringWordFeatures(DNA)
                feats_test_k1.obtain_from_char(charfeat_test, i-1, i, GAP, False)
                feats_test_k1.add_preprocessor(preproc)
                feats_test_k1.apply_preprocessor()
                # append features
                feats_train.append_feature_obj(charfeat_train);
                feats_test.append_feature_obj(charfeat_test);
		# append spectrum kernel
                kernel1=CommWordStringKernel(10,i);
                kernel1.io.set_loglevel(MSG_DEBUG);
                kernel.append_kernel(kernel1);

	'''
	Uncomment this for Multiple Weighted degree kernels and comment
	the multiple spectrum kernel block above instead

	##################### Multiple Weighted Degree Kernel #########################
	for i in range(K1,K2,-1):
		# append training data to combined feature object
		charfeat_train = StringCharFeatures(list(train_xt), DNA)
		# append testing data to combined feature object
		charfeat_test = StringCharFeatures(test_xt, DNA)
		# append features
		feats_train.append_feature_obj(charfeat_train);
    		feats_test.append_feature_obj(charfeat_test);
		# setup weighted degree kernel		
		kernel1=WeightedDegreePositionStringKernel(10,i);
    		kernel1.io.set_loglevel(MSG_DEBUG);
		kernel1.set_shifts(SHIFT*np.ones(len(train_xt[0]), dtype=np.int32))
		kernel1.set_position_weights(np.ones(len(train_xt[0]), dtype=np.float64));
		kernel.append_kernel(kernel1);
	'''

	##################### Training #########################

	print "Starting MKL training.."
	mkl = MKLClassification();
	mkl.set_mkl_norm(3) #1,2,3
	mkl.set_C(SVMC, SVMC)
	mkl.set_kernel(kernel)
	mkl.set_labels(labels)
	mkl.train(feats_train)
	
	print "Making predictions!"
	out1 = mkl.apply(feats_train).get_labels();
	out2 = mkl.apply(feats_test).get_labels();

	return out1,out2,train_lt

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


if __name__=='__main__':
	print('LibSVR')
	train_xt = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeIntList(TRAININGLABELSFILENAME)
	test_xt = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeIntList(VALIDATIONLABELSFILENAME)
	[out1, out2, train_lt] = runShogunSVMMultipleKernels(train_xt, train_lt, test_xt)
	outputResultsClassification(out1, out2, train_lt, test_lt)

