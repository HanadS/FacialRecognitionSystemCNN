import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import math
from sklearn.datasets import fetch_lfw_people

########################################################################
# Load the data as numpy arrays. #

faces = fetch_lfw_people(color=False, min_faces_per_person=10)


########################################################################
# Split into a training set and testing. #

X = faces.data
y = faces.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42) 
    
numTrain = X_train.shape[0]
numTest = X_test.shape[0]

X_train = X_train.reshape(numTrain, 62, 47)
X_test = X_test.reshape(numTest, 62, 47)

# data dimensions. #
h, w = X_train[0].shape

X_train = X_train.reshape(numTrain, h*w)
X_test = X_test.reshape(numTest, h*w)

########################################################################
# Calculate mean face. #

meanFace = np.zeros(X_train[0].shape)
for i in range(numTrain):
    meanFace = meanFace + X_train[0]

for i in range((h*w)):
    meanFace[i] = meanFace[i] / (numTrain)

########################################################################
# Normalize the training set. #

X_train_normalized = X_train
for i in range(numTrain):
    X_train_normalized[i] = X_train_normalized[i] - meanFace

########################################################################
# Calculate the covariance. #

X_train_normalized_T = np.transpose(X_train_normalized)

covariance = np.dot(X_train_normalized, X_train_normalized_T)

########################################################################
# Compute a PCA and produce eigenfaces. #

numComponents = 150

pca = PCA(n_components=numComponents, svd_solver='randomized',
          whiten=True).fit(X_train_normalized)

eigenfaces = pca.components_.reshape((numComponents, (h*w)))


########################################################################
# Calculate weights for the training set for each eigenvector. #

trainWeights = np.zeros((numTrain, 150))

for i in range(numTrain):
    trainFace = X_train[i]
    weights = np.zeros((150, ))
    
    for j in range(150):
        weights[j] = np.dot(trainFace, eigenfaces[j])
    
    trainWeights[i] = weights

########################################################################
# Recognition. #

numberOfSuccesses = 0
numberOfFailures = 0
threshold = 490 # empirically driven threshold choice.

# K-Fold. #
fold = np.array_split(range(X_test.shape[0]), 256)            

testNumber = 50
#for f in fold:
for i in range(testNumber):
    newFace = X_test[i] - meanFace
    newFaceWeights = np.zeros((150, ))
    
    for j in range(150):
        newFaceWeights[j] = np.dot(newFace, eigenfaces[j])

    # set the currentlowest distance to infinity
    currentLowestDistance = float('inf')
    guesses = set()
    
    for k in range(numTrain):
        dist = np.sqrt(np.linalg.norm(abs(newFaceWeights - trainWeights)))
        if dist < threshold:
            currentLowestDistance = dist
            guesses.add(y_train[k])
            
    # Is the correct answer among our candidate faces. #
    if y_test[i] in guesses:
        numberOfSuccesses = numberOfSuccesses + 1
    else:
        numberOfFailures = numberOfFailures + 1
        
    print("Test Image %d.\n" % (i + 1))
        
print("Successes: %d. Failures: %d. Accuracy: %.2f%%." % (numberOfSuccesses, numberOfFailures, (numberOfSuccesses/testNumber) * 100))

    

