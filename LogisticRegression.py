import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LogisticRegression:

    # Values based on multiple experiments on the model
    epochs = 10000
    learn_rate = 0.2
    bias = 0

    # Method to find the sigmoid to get the value of Target in range [0,1]
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    # Method to implement Model of Logistic Regression from scratch based on Mathematical formula
    def model(self,X,Y):
        # m is No. of observations
        # n is No. of Features
        m = X.shape[1]
        n= X.shape[0]

        # Initializing the weights with Random Value Array of size equal to no. of features
        weights = np.random.randn(X.shape[0], 1) * 0.01

        # Loss List and Accuracy List are placed to find how Cost Reduced and Accuracy Improved over the Iterations onto Plot
        loss_list = []
        accuracy_list = []

        # Initializing Bias and Learning Rate which are declared in Global
        bias, learn_rate = self.bias,self.learn_rate

        # On every Iteration of epochs the weights and bias will be adjusted in order Minimize the Loss Function that is Gradient Descent
        for i in range(self.epochs):

            Z = np.dot(weights.T, X) + bias
            # Predicted Y with Sigmoid Functio
            Y_pred = self.sigmoid(Z)

            # Cost Function
            cost = -(1 / m) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))

            # Gradients of weights and bias
            dW = (1 / m) * np.dot(Y_pred - Y, X.T)
            dB = (1 / m) * np.sum(Y_pred - Y)

            # Optimizing weights and bias with Gradient, learning rate and previous values
            weights = weights - learn_rate * dW.T
            bias = bias - learn_rate * dB

            # Converting the Prediction Matrix to 0's and 1's on basis of threshold 0.5
            Y_pred = Y_pred > 0.5
            Y_pred = np.array(Y_pred, dtype='int64')
            accu = (1 - np.sum(np.absolute(Y_pred - Y)) / Y.shape[1])

            # Adding Loss and Accuracy on each Iteration for Graph Plotting
            loss_list.append(cost)
            accuracy_list.append(accu)

            # Printing Loss and Accuracy 10 time over the course of iterations to check improvements
            if(i%(self.epochs/10) == 0):
                print("Loss and Accuracy after ",i," iteration are : ",cost," and ",accu," respectively.")
        return weights,bias,loss_list,accuracy_list

    # Method to find the Accuracy of Various Datasets based on Trained Model
    def accuracy(self,X, Y, W, B):

        # Calculating the Target from given weights and bias
        Z = np.dot(W.T, X) + B
        Y_pred = self.sigmoid(Z)

        Y_pred = Y_pred > 0.5
        Y_pred = np.array(Y_pred, dtype='int64')
        acc = (1 - np.sum(np.absolute(Y_pred - Y)) / Y.shape[1]) * 100

        return acc

# Fetching the Data of Csv into Panda DataFrame
classification_data = pd.read_csv('diabetes.csv')

# Splitting Data into X=Features and Y=Target
X = classification_data.drop(columns="Outcome")
Y = classification_data["Outcome"]

# Splitting Features and Targets to Train,Validation and Test Datasets in 0.6:0.2:0.2 Ratio
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,train_size = 0.8)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=0.25)

# Normalizing the Features of Train and Test Datasets to better Model the Logistic Regression with best Accuracy
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

# Operation on Dataframe and Array DataSets to convert to Array and Reshaping them to get them Dot Matrix Multiplied and other Operations
X_train,Y_train = X_train.T,Y_train.values.reshape(1,Y_train.shape[0])
X_validation,Y_validation = X_validation.T,Y_validation.values.reshape(1,Y_validation.shape[0])
X_test,Y_test = X_test.T,Y_test.values.reshape(1,Y_test.shape[0])

# Initializing the object of class to access the Methods for Appropriate Operations
Lr = LogisticRegression()
print("===================== Part 1 - Logistic Regression Classicfication =====================\n\n")
w,b,loss_list,accuracy_list = Lr.model(X_train,Y_train)

# Fetching the Accuracy of Training, Validation and Test Dataset after training the Model
acc_training = Lr.accuracy(X_train,Y_train,w,b)
acc_validation = Lr.accuracy(X_validation,Y_validation,w,b)
acc_test = Lr.accuracy(X_test,Y_test,w,b)

# Displaying the Accuracy in Percentage
print("Accuracy of Training Dataset on basis of trained  model is : ", round(acc_training, 2), "%")
print("Accuracy of Validation Dataset on basis of trained  model is : ", round(acc_validation, 2), "%")
print("Accuracy of Test Dataset on basis of trained  model is : ", round(acc_test, 2), "%")

#Plotting the Train vs Validation Loss
plt.plot(loss_list)
#Plotting the Train vs Validation Accuracy
plt.plot(accuracy_list)
plt.show()

