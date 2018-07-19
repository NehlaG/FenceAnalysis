from __future__ import print_function
import cv2
import glob
import numpy as np
import math
from time import time
import sys, os
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from utilities import processArguments
params = {
'method': 0,
}


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

processArguments(sys.argv[1:], params)
method = params['method']
if method == 0:
    print('Using scikit-learn API for transformation')
elif method == 1:
    print('Using matrix operations for transformation')
# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

X_data = []

files = glob.glob ("/home/nehla/scikit_learn_data/lfw_home/lfw_funneled/George_Clooney/*.jpg")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile).flatten()
    X_data.append (image)

#print('X_data shape:', np.array(X_data).shape)
n_samples, n_features=np.array(X_data).shape
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" %n_features)
#print("n_classes: %d" % n_classes)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, n_samples))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(X_data)
print("done in %0.3fs" % (time() - t0))

print ("Principal axes in feature space, representing the directions of maximum variance in the data ")
#eigenfaces = pca.components_.reshape((n_components, h, w))

print ("Saving PCA components into file... ")
np.savetxt('pca-components.txt',  np.array(pca.components_ ), fmt='%f')
# #############################################################################
# Compute the construction error
X_test= []
#myFile='/home/nehla/scikit_learn_data/lfw_home/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
myFile='/home/nehla/scikit_learn_data/lfw_home/lfw_funneled/George_Clooney/George_Clooney_0002.jpg'
image = cv2.imread (myFile).flatten()
X_test.append (image)
n_samples_test, n_features_test=np.array(X_test).shape
print('X_test shape:', np.array(X_test).shape)

print('components_ shape:', np.array(pca.components_ ).shape)

print("Projecting the test data on the eigenfaces orthonormal basis")

if method == 0:
    X_test_pca = pca.transform(X_test)
else :
    X_test_pca=np.dot(X_test,np.transpose(pca.components_)) ## second implementation

print(' transformed test data shape:', np.array(X_test_pca).shape)

print("Transform test data back to its original space")

if method == 0:
    X_test_reprojected= pca.inverse_transform(X_test_pca )
else :
    X_test_reprojected=np.dot(X_test_pca, pca.components_) ## second implementation

#  error = L2_norm(T - T'')
l2_norm_error=np.linalg.norm (np.array(X_test)-X_test_reprojected)
print("l2_norm_error : %f" %l2_norm_error)
