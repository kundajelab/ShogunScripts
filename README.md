# ShogunScripts
These scripts were written by Irene Kaplow for running Shogun with various kernels.
The following scripts involve kernels that do not account for positional information: runShogunSVMDNALinearStringKernel.py, runShogunSVMDNALocalAlignmentKernel.py, runShogunSVMDNAOligoStringKernel.py, runShogunSVMDNASpectrumKernel.py, runShogunSVMDNASubsequenceStringKernel.py, runShogunSVMDNAUlongSpectrumKernel.py, runShogunSVMDNAWeightedCommonWordKernel.py, and runSogunSVRSpectrumKernel.py.
The following scripts involve kernels that do account for positional information: runShogunSVMDNAFixedDegreeKernel.py, runShogunSVMDNAWDKernel.py, runShogunSVMDNANoPositionKernel.py (this is a WD kernel without shifts), and runShogunSVRWDKernel.py.
Every script with "SVM" in its name is for an SVM; every script with "SVR" in its name is for an SVR.
Every script with "DNA" in its name is for a DNA alphabet; all other scripts are for protein alphabets.
The following scripts involve SVMs that are extremly slow to train: runShogunSVMDNALinearStringKernel.py, runShogunSVMDNALocalAlignmentKernel.py, runShogunSVMDNAOligoStringKernel.py, and runShogunSVMDNASubsequenceStringKernel.py.
All other SVMs should train in less than half a day on 10,000 examples with k/degree <= 11.
