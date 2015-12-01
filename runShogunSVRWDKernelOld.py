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
FNEPSILON = sys.argv[6]
FNNU = sys.argv[7]
TRAINPREDICTIONSEPSILONFILENAME = sys.argv[8]
VALIDATIONPREDICTIONSEPSILONFILENAME = sys.argv[9]
TRAINPREDICTIONSNUFILENAME = sys.argv[10]
VALIDATIONPREDICTIONSNUFILENAME = sys.argv[11]
KERNELFILENAME = sys.argv[12]
SIZE = 100


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


def get_wd_features(data, feat_type="protein"):
	"""
	create feature object for wdk
	"""
	if feat_type == "dna":
		feat = StringCharFeatures(DNA)
	elif feat_type == "protein":
		feat = StringCharFeatures(PROTEIN)
	else:
		raise Exception("unknown feature type")
	feat.set_features(data)

	return feat
	
	
def get_spectrum_features(data, order=3, gap=0, reverse=True):
	"""
	create feature object used by spectrum kernel
	"""

	charfeat = StringCharFeatures(data, PROTEIN)
	feat = StringWordFeatures(charfeat.get_alphabet())
	feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc = SortWordString()
	preproc.init(feat)
	feat.add_preprocessor(preproc)
	feat.apply_preprocessor()

	return feat
	
	
def construct_features(features):
	"""
	makes a list
	"""

	feat_all = [inst for inst in features]
	feat_lhs = [inst[0:15] for inst in features]
	feat_rhs = [inst[15:] for inst in features]

	feat_wd = get_wd_features(feat_all)
	feat_spec_1 = get_spectrum_features(feat_all, order=3)
	#feat_spec_1 = get_spectrum_features(feat_lhs, order=3)
	#feat_spec_2 = get_spectrum_features(feat_rhs, order=3)

	feat_comb = CombinedFeatures()
	feat_comb.append_feature_obj(feat_wd)
	feat_comb.append_feature_obj(feat_spec_1)
	#feat_comb.append_feature_obj(feat_spec_2)

	return feat_comb
	
	
def runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1, svr_param=0.1):
	"""
	serialize svr with string kernels
	"""

    ##################################################
    # set up svr
	feats_train = construct_features(train_xt)
	feats_test = construct_features(test_xt)

	max_len = len(train_xt[0])
	kernel_wdk = WeightedDegreePositionStringKernel(SIZE, 5)
	shifts_vector = np.ones(max_len, dtype=np.int32)*NUMSHIFTS
	kernel_wdk.set_shifts(shifts_vector)

    ########
    # set up spectrum
	use_sign = False
	kernel_spec_1 = WeightedCommWordStringKernel(SIZE, use_sign)
	#kernel_spec_2 = WeightedCommWordStringKernel(SIZE, use_sign)

    ########
    # combined kernel
	kernel = CombinedKernel()
	kernel.append_kernel(kernel_wdk)
	kernel.append_kernel(kernel_spec_1)
	#kernel.append_kernel(kernel_spec_2)

    # init kernel
	labels = RegressionLabels(train_lt)
	
	# two svr models: epsilon and nu
	svr_epsilon=LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_EPSILON_SVR)
	print "Ready to train!"
	svr_epsilon.train(feats_train)
	#svr_nu=LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_NU_SVR)
	#svr_nu.train(feats_train)

	# predictions
	print "Making predictions!"
	kernel.init(feats_train, feats_test)
	out1_epsilon=svr_epsilon.apply().get_labels()
	out2_epsilon=svr_epsilon.apply(feats_test).get_labels()
	#out1_nu=svr_epsilon.apply().get_labels()
	#out2_nu=svr_epsilon.apply(feats_test).get_labels()

    ##################################################
    # serialize to file
	fEpsilon = open(FNEPSILON, 'w+')
	#fNu = open(FNNU, 'w+')
	svr_epsilon.save(fEpsilon)
	#svr_nu.save(fNu)
	fEpsilon.close()
	#fNu.close()

    ##################################################
	
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
	kernel.save(KERNELFILENAME)


if __name__=='__main__':
	print('LibSVR')
	train_xt = makeStringList(TRAININGDATAFILENAME)
	train_lt = makeFloatList(TRAININGLABELSFILENAME)
	test_xt = makeStringList(VALIDATIONDATAFILENAME)
	test_lt = makeFloatList(VALIDATIONLABELSFILENAME)
	#[out1_epsilon, out2_epsilon, out1_nu, out2_nu, kernel] = runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1, svr_param=0.1)
	[out1_epsilon, out2_epsilon, kernel] = runShogunSVRWDKernel(train_xt, train_lt, test_xt, svm_c=1, svr_param=0.1)
	outputResults(out1_epsilon, out2_epsilon, out1_nu, out2_nu, kernel, train_lt, test_lt)