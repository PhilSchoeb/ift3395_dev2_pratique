import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    """
    y : numpy array of shape (n,)
    m : int (num_classes)
    returns : numpy array of shape (n, m) for one versus all
    """
    def make_one_versus_all_labels(self, y, m):
        n = len(y)
        return_array = []
        for i in range(n):
            add_array = []
            for j in range(m):
                if y[i] == j:
                    add_array.append(1)
                else:
                    add_array.append(-1)
            return_array.append(add_array)
        return np.array(return_array)


    def indicator(self, label, y):
        if label == np.argmax(y):
            return 1
        else:
            return -1

    # x is a single column of attributes and y a single label vector
    def loss(self, label, x, y):
        return (max(0, 2 - np.dot(np.transpose(self.w[:, label]), x) * self.indicator(label, y)))**2

    """
    x : numpy array of shape (minibatch size, num_features)
    y : numpy array of shape (minibatch size, num_classes)
    returns : float
    """
    def compute_loss(self, x, y):
        n = len(x)
        m = len(y[0])
        loss = 0
        for i in range(n):
            for j in range(m):
                loss += self.loss(j, np.transpose(x[i]), y[i])
        reg = 0
        for i in range(m):
            reg += (np.linalg.norm(self.w[:, i]))**2
        loss += (self.C / 2) * reg
        return loss

    def f_minus(self, label, x, d):
        if 2. - np.dot(np.transpose(self.w[:, label]), x) <= 0.:
            return 0
        else:
            return (2. - np.dot(np.transpose(self.w[:, label]), x)) * (-x[d])

    def f_plus(self, label, x, d):
        if 2. + np.dot(np.transpose(self.w[:, label]), x) <= 0.:
            return 0
        else:
            return (2. + np.dot(np.transpose(self.w[:, label]), x)) * x[d]

    """
    x : numpy array of shape (minibatch size, num_features)
    y : numpy array of shape (minibatch size, num_classes)
    returns : numpy array of shape (num_features, num_classes)
    """
    def compute_gradient(self, x, y):
        n = len(x)
        d = len(x[0])
        m = len(y[0])
        gradient = []
        for i in range(d):  # Attribute
            grad_line = []
            for j in range(m):  # Label
                sum = 0
                for k in range(n):  # Point
                    if j == np.argmax(y[k]):
                        sum += self.f_minus(j, np.transpose(x[k]), i)
                    else:
                        sum += self.f_plus(j, np.transpose(x[k]), i)
                grad = (2 / n) * sum
                grad_reg = self.C * self.w[i, j]
                grad += grad_reg
                grad_line.append(grad)
            gradient.append(grad_line)
        return np.array(gradient)

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size

        for ndx in range(0, l, n):

            index2 = min(ndx + n, l)

            yield iterable1[ndx: index2], iterable2[ndx: index2]

    """
    x : numpy array of shape (num_examples_to_infer, num_features)
    returns : numpy array of shape (num_examples_to_infer, num_classes)
    """
    def infer(self, x):
        n = len(x)
        preds = []
        for i in range(n):
            scores = []
            for j in range(self.m):
                score = np.dot(np.transpose(self.w[:, j]), np.transpose(x[i]))
                scores.append(score)
            #print(scores)
            index_max = np.argmax(scores)
            scores[index_max] = 1
            for j in range(self.m):
                if j != index_max:
                    scores[j] = -1
            preds.append(scores)
        #print(preds[15:50])
        return np.array(preds)

    """
    y_inferred : numpy array of shape (num_examples, num_classes)
    y : numpy array of shape (num_examples, num_classes)
    returns : float
    """
    def compute_accuracy(self, y_inferred, y):
        good_predic = 0
        n = len(y)
        for i in range(n):
            prediction = np.argmax(y_inferred[i])
            actual_label = np.argmax(y[i])
            if prediction == actual_label:
                good_predic += 1
        return float(good_predic) / n


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
        #self.w = np.full([self.num_features, self.m], 0.025, dtype=float)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        #print(self.w)
        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            #print(self.w)
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
    svm = SVM(eta=0.0001, C=1, niter=200, batch_size=100, verbose=True)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 3) # one-versus-all labels
    # svm.w = np.zeros([x_train.shape[1], 3])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
