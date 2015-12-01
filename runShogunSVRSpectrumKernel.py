# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/serialization_string_kernels_modular.py

import numpy as np
from modshogun import RegressionLabels, RealFeatures
from modshogun import LibSVR, LIBSVR_NU_SVR, LIBSVR_EPSILON_SVR
from modshogun import StringCharFeatures, RealFeatures, CombinedFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, PROTEIN, Labels
from modshogun import MSG_DEBUG
from modshogun import SVMLight
from modshogun import CommWordStringKernel
from modshogun import SortWordString
from numpy import concatenate, ones
from numpy.random import randn, seed
from numpy import zeros,ones,float64,int32
import sys
import types
import random
import bz2
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr


TRAININGDATAFILENAME = sys.argv[1]
TRAININGLABELSFILENAME = sys.argv[2]
VALIDATIONDATAFILENAME = sys.argv[3]
VALIDATIONLABELSFILENAME = sys.argv[4]
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[5]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[6]
K = int(sys.argv[7])
SVRPARAM = float(sys.argv[8]) # Initially 0.1


def makeStringList(stringFileName):
	# Get a string list from a file
	stringList = []
	stringFile = open(stringFileName)
	
	for line in stringFile:
		# Iterate through the string file and get the string from each line
		stringList.append(line.strip())
		
	stringFile.close()
	return stringList


def makeFloatList(floatFileName):
	# Get a float list from a file
	floatList = []
	floatFile = open(floatFileName)
	
	for line in floatFile:
		# Iterate through the float file and get the float from each line
		floatList.append(float(line.strip()))
		
	floatFile.close()
	return np.array(floatList)
	
	
def runShogunSVRSpectrumKernel(train_xt, train_lt, test_xt, svm_c=1):
	"""
	serialize svr with spectrum kernels
	"""

    ##################################################
    # set up svr
	charfeat_train = StringCharFeatures(train_xt, PROTEIN)
	feats_train = StringWordFeatures(PROTEIN)
	feats_train.obtain_from_char(charfeat_train, K-1, K, 0, False)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()
	
	charfeat_test = StringCharFeatures(test_xt, PROTEIN)
	feats_test=StringWordFeatures(PROTEIN)
	feats_test.obtain_from_char(charfeat_test, K-1, K, 0, False)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()
	
	kernel=CommWordStringKernel(feats_train, feats_train, False)
	kernel.io.set_loglevel(MSG_DEBUG)

    # init kernel
	labels = RegressionLabels(train_lt)
	
	# two svr models: epsilon and nu
	print "Ready to train!"
	svr_epsilon=LibSVR(svm_c, SVRPARAM, kernel, labels, LIBSVR_EPSILON_SVR)
	svr_epsilon.io.set_loglevel(MSG_DEBUG)
	svr_epsilon.train()

	# predictions
	print "Making predictions!"
	out1_epsilon=svr_epsilon.apply(feats_train).get_labels()
	kernel.init(feats_train, feats_test)
	out2_epsilon=svr_epsilon.apply(feats_test).get_labels()

	return out1_epsilon,out2_epsilon,kernel


def writeFloatList(floatList, floatListFileName):
	# Write a list of floats to a file
	floatListFile = open(floatListFileName, 'w+')
	
	for f in floatList:
		# Iterate through the floats and record each of them
		floatListFile.write(str(f) + "\n")
	floatListFile.close()


def outputResults(out1_epsilon, out2_epsilon, kernel,  train_lt, test_lt):
	# Output the results to the appropriate output files
	writeFloatList(out1_epsilon, TRAINPREDICTIONSEPSILONFILENAME)
	writeFloatList(out2_epsilon, VALIDATIONPREDICTIONSEPSILONFILENAME)
	print "Pearson correlation between training labels and predictions, epsilon SVR:"
	print pearsonr(train_lt, out1_epsilon)
	print "Spearman correlation between training labels and predictions, epsilon SVR:"
	print spearmanr(train_lt, out1_epsilon)
	print "Pearson correlation between validation labels and predictions, epsilon SVR:"
	print pearsonr(test_lt, out2_epsilon)
	print "Spearman correlation between validation labels and predictions, epsilon SVR:"
	print spearmanr(test_lt, out2_epsilon)


if __name__=='__main__':
	print('LibSVR')
	train_xt = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeFloatList(TRAININGLABELSFILENAME)
	test_xt = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeFloatList(VALIDATIONLABELSFILENAME)
	[out1_epsilon, out2_epsilon, kernel] = runShogunSVRSpectrumKernel(train_xt, train_lt, test_xt, svm_c=1)
	outputResults(out1_epsilon, out2_epsilon, kernel, train_lt, test_lt)