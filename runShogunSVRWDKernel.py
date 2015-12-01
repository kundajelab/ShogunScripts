# Original code from: ../examples/documented/python_modular/regression_libsvr_modular.py
# and: ../examples/documented/python_modular/serialization_string_kernels_modular.py

import numpy as np
from modshogun import RegressionLabels, RealFeatures
from modshogun import GaussianKernel
from modshogun import LibSVR, LIBSVR_NU_SVR, LIBSVR_EPSILON_SVR
from modshogun import WeightedDegreeStringKernel, LinearKernel, PolyKernel, GaussianKernel, CTaxonomy
from modshogun import CombinedKernel, WeightedDegreeRBFKernel
from modshogun import StringCharFeatures, RealFeatures, CombinedFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, PROTEIN, Labels
from modshogun import WeightedDegreeStringKernel, CombinedKernel, WeightedCommWordStringKernel, WeightedDegreePositionStringKernel
from modshogun import StringCharFeatures, DNA, StringWordFeatures, CombinedFeatures
from modshogun import MSG_DEBUG
from modshogun import RealFeatures, BinaryLabels, DNA, Alphabet
from modshogun import WeightedDegreeStringKernel, GaussianKernel
from modshogun import SVMLight
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
NUMSHIFTS = int(sys.argv[5])
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[6]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[7]
TRAINPREDICTIONSNUFILENAME = sys.argv[8]
VALIDATIONPREDICTIONSNUFILENAME = sys.argv[9]
KERNELFILENAME = sys.argv[10]
DEGREE = int(sys.argv[11])
SVRPARAM = float(sys.argv[12]) # Initially 0.1


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
	
	
def runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1):
	"""
	serialize svr with string kernels
	"""

    ##################################################
    # set up svr
	feats_train = StringCharFeatures(train_xt, PROTEIN)
	feats_test = StringCharFeatures(test_xt, PROTEIN)

	kernel = WeightedDegreePositionStringKernel(feats_train, feats_train, DEGREE)
	kernel.io.set_loglevel(MSG_DEBUG)
	kernel.set_shifts(NUMSHIFTS*ones(len(train_xt[0]), dtype=int32))
	kernel.set_position_weights(ones(len(train_xt[0]), dtype=float64))

    # init kernel
	labels = RegressionLabels(train_lt)
	
	# two svr models: epsilon and nu
	print "Ready to train!"
	svr_epsilon=LibSVR(svm_c, SVRPARAM, kernel, labels, LIBSVR_EPSILON_SVR)
	svr_epsilon.io.set_loglevel(MSG_DEBUG)
	svr_epsilon.train()
	#svr_nu=LibSVR(svm_c, SVRPARAM, kernel, labels, LIBSVR_NU_SVR)
	#svr_nu.train()

	# predictions
	print "Making predictions!"
	out1_epsilon=svr_epsilon.apply(feats_train).get_labels()
	kernel.init(feats_train, feats_test)
	out2_epsilon=svr_epsilon.apply(feats_test).get_labels()
	#out1_nu=svr_epsilon.apply(feats_train).get_labels()
	#out2_nu=svr_epsilon.apply(feats_test).get_labels()
	
	#return out1_epsilon,out2_epsilon,out1_nu,out2_nu ,kernel
	return out1_epsilon,out2_epsilon,kernel


def writeFloatList(floatList, floatListFileName):
	# Write a list of floats to a file
	floatListFile = open(floatListFileName, 'w+')
	
	for f in floatList:
		# Iterate through the floats and record each of them
		floatListFile.write(str(f) + "\n")
	floatListFile.close()


#def outputResults(out1_epsilon, out2_epsilon, out1_nu, out2_nu, kernel,  train_lt, test_lt):
def outputResults(out1_epsilon, out2_epsilon, kernel,  train_lt, test_lt):
	# Output the results to the appropriate output files
	writeFloatList(out1_epsilon, TRAINPREDICTIONSEPSILONFILENAME)
	writeFloatList(out2_epsilon, VALIDATIONPREDICTIONSEPSILONFILENAME)
	#writeFloatList(out1_nu, TRAINPREDICTIONSNUFILENAME)
	#writeFloatList(out2_nu, VALIDATIONPREDICTIONSNUFILENAME)
	print "Pearson correlation between training labels and predictions, epsilon SVR:"
	print pearsonr(train_lt, out1_epsilon)
	print "Spearman correlation between training labels and predictions, epsilon SVR:"
	print spearmanr(train_lt, out1_epsilon)
	print "Pearson correlation between validation labels and predictions, epsilon SVR:"
	print pearsonr(test_lt, out2_epsilon)
	print "Spearman correlation between validation labels and predictions, epsilon SVR:"
	print spearmanr(test_lt, out2_epsilon)
	#print "Pearson correlation between training labels and predictions, nu SVR:"
	#print pearsonr(train_lt, out1_nu)
	#print "Spearman correlation between training labels and predictions, nu SVR:"
	#print spearmanr(train_lt, out1_nu)
	#print "Pearson correlation between validation labels and predictions, nu SVR:"
	#print pearsonr(test_lt, out2_nu)
	#print "Spearman correlation between validation labels and predictions, nu SVR:"
	#print spearmanr(test_lt, out2_nu)


if __name__=='__main__':
	print('LibSVR')
	train_xt = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeFloatList(TRAININGLABELSFILENAME)
	test_xt = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeFloatList(VALIDATIONLABELSFILENAME)
	#[out1_epsilon, out2_epsilon, out1_nu, out2_nu, kernel] = runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1)
	[out1_epsilon, out2_epsilon, kernel] = runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1)
	#outputResults(out1_epsilon, out2_epsilon, out1_nu, out2_nu, kernel, train_lt, test_lt)
	outputResults(out1_epsilon, out2_epsilon, kernel, train_lt, test_lt)