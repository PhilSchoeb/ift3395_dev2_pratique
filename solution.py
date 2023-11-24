
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        
        resultat = np.zeros((len(y), m))

        for i in range(len(y)):
            resultat[i, y[i]] = 1
            resultat[i, np.arange(m) != y[i]] = -1

        return resultat

    def compute_loss(self, x, y):
        num_examples = len(x)
        loss = 0.0
        #w = np.zeros((len(x[0]),len(y[0])))
        w=self.w
        
        for i in range(num_examples):
            for j in range(len(y[0])):
                indicator = 1 if y[i, j] == j else -1
                hinge_loss = max(0, 2 - np.dot(w[:,j], x[i])*indicator) ** 2
                
                loss += hinge_loss


        termeRegu=0
        for i in range(len(y[0])):
            termeRegu+=np.dot(w[:,i],w[:,i])


        return (loss / num_examples)+(self.C/2)*termeRegu

    def indicator(self, label, y):
        if label == np.argmax(y):
            return 1
        else:
            return -1
        

    def f_minus(self, label, x, d):
        margin = 2. - np.dot(self.w[:, label], x)
        return max(0, margin) * (-x[d])

    def f_plus(self, label, x, d):
        margin = 2. + np.dot(self.w[:, label], x)
        return max(0, margin) * x[d]

    def compute_gradient(self, x, y):
        n, d = x.shape
        _, m = y.shape

        gradient = np.zeros((d, m))

        for i in range(d):  # Attribute
            for j in range(m):  # Label
                sum_term = 0
                for k in range(n):  # Point
                    indicator = 1 if j == np.argmax(y[k]) else -1
                    sum_term += self.f_minus(j, x[k], i) if indicator == 1 else self.f_plus(j, x[k], i)

                grad = (2 / n) * sum_term + self.C * self.w[i, j]
                gradient[i, j] = grad

        return gradient




    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size

        for ndx in range(0, l, n):

            index2 = min(ndx + n, l)

            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        n, _ = x.shape
        _, m = self.w.shape

        y_inferred = np.zeros((n, 6)) - 1  # start all at -1

        for i in range(n):
            scoresClasses = np.zeros(m)  # Initialize outside the loop
            for j in range(m):
                scores = np.dot(self.w[:, j], x[i])
                scoresClasses[j] = scores

            predicted_class = np.argmax(scoresClasses)
            y_inferred[i, predicted_class] = 1
            #print(y_inferred[i,:])

        return y_inferred

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        vraiClasses= np.sum(np.argmax(y_inferred, axis=1) == np.argmax(y, axis=1))
        total_examples = len(y)
        accuracy = vraiClasses / total_examples

        return accuracy

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Star_classification/"
    dataset = pd.read_csv(data_path + "star_classification.csv")
    y = dataset['class']
    x = dataset.drop(['class','rerun_ID'], axis=1)
    
    #we replace the dataset class with a number (the class are : 'GALAXY' 'QSO' 'STAR')
    y = y.replace('GALAXY', 0)
    y = y.replace('QSO', 1)
    y = y.replace('STAR', 2)

    #split dataset in train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=40)

    #convert sets to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test=np.array(y_test)

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=1, niter=3, batch_size=100, verbose=False)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(y_train, 3) # one-versus-all labels
    svm.w = np.zeros([x_train.shape[1], 3])
    grad = svm.compute_gradient(x_train, y_train_ova)
    loss = svm.compute_loss(x_train, y_train_ova)

    print(train_losses)
    print(train_accs)
    print(test_losses)
    print(test_accs)