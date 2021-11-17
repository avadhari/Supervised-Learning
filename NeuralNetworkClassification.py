from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2,l1,l1_l2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class NeuralNetworkClassifier:

    # Method of Classification which trains with Neural Network with Either L2 Regularization or Dropout Regularizaer
    def classification_model(self,X,Y,X_val,Y_val,n1,n2,iters,batch,learn_rate,lambd,dropout,d1 = 0,d2=0):

        # Defining Model as Sequential
        model = Sequential()

        # If true is passed in parameter of Dropout we implement with Dropout or else L2 Regularizer
        if dropout :
            model.add(Dense(n1, input_dim=8, activation='LeakyReLU')) # Hidden Layer 1
            model.add(Dropout(d1))
            model.add(Dense(n2, activation='LeakyReLU')) # Hidden Layer 2
            model.add(Dropout(d2))
        else :
          model.add(Dense(n1, input_dim=8, activation='LeakyReLU', activity_regularizer=l2(lambd))) # Hidden Layer 1
          model.add(Dense(n2, activation='LeakyReLU')) # Hidden Layer 2

        #  Output Layer is same for both and has Sigmoid as activation
        model.add(Dense(1, activation='sigmoid'))

        # We use Adam as Optimizer for our Model to train with learning rate 0.002
        opt = Adam(learning_rate=learn_rate)

        # Compiling the Model with Loss Functions as Binary Cross Entropy as the Classification of Model is Binary
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Atlast Fitting the Training Dataset into the Model with specified Epochs(Iterations) and Batches
        history = model.fit(X, Y, validation_data= (X_val,Y_val) ,epochs=iters, batch_size=batch, verbose=0)

        return model,history

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

# Creating the Object of Classifier Class to call the Classification Model Method
NNclassifier = NeuralNetworkClassifier()

#Fetching the Model with Regularizar as L2
model_part2, history = NNclassifier.classification_model(X_train,Y_train,X_validation,Y_validation,12,8,198,16,0.002,0.004,False)

print('===================== Part 2 - Accuracies with L2(Ridge) Regularizer ===================== \n')
_,accuracy = model_part2.evaluate(X_train,Y_train)
print('Accuracy of Training Dataset: %.2f' % (accuracy*100))

_,accuracy =model_part2.evaluate(X_validation,Y_validation)
print('Accuracy of Validation Dataset: %.2f' % (accuracy*100))

_,accuracy = model_part2.evaluate(X_test,Y_test)
print('Accuracy of Testing Dataset: %.2f' % (accuracy*100))

# Accuracy plots for validation and train data
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Accuracy of Validation and Train as per Epochs')

ax1.plot(history.history['val_accuracy'])
ax1.set_ylabel('Validation-Accuracy')
ax1.set_xlabel('Epochs')

ax2.plot(history.history['accuracy'])
ax2.set_ylabel('Train Accuracy')
ax2.set_xlabel('Epochs')

plt.show()

# Loss plot for validation and train data
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Loss of Validation and Train as per Epochs')

ax1.plot(history.history['val_loss'])
ax1.set_ylabel('Validation Loss')
ax1.set_xlabel('Epochs')

ax2.plot(history.history['loss'])
ax2.set_ylabel('Train Loss')
ax2.set_xlabel('Epochs')
plt.show()

print('\n\n===================== Part 3 - Comparing Regularizer Accuracies for different Neural Network Models (L2 & Dropout) ===================== \n')

#Fetching the Model with Regularizar as L2 for comparing with the Dropout Regularizer
model_regularizer_part3,_ = NNclassifier.classification_model(X_train,Y_train,X_validation,Y_validation,20,12,200,16,0.002,0.004,False)

#Fetching the Model with Regularizar as Dropout for comparing with the L2 Regularizer
model_dropout_part3,_ = NNclassifier.classification_model(X_train,Y_train,X_validation,Y_validation,20,12,200,16,0.002,0.004,True,0.4,0.34)

# Evaluating the Different Accuracies of two different Model of Neural Network
_,accuracy_train_reg = model_regularizer_part3.evaluate(X_train,Y_train)
_,accuracy_validation_reg = model_regularizer_part3.evaluate(X_validation,Y_validation)
_,accuracy_test_reg = model_regularizer_part3.evaluate(X_test,Y_test)
_,accuracy_train_drop = model_dropout_part3.evaluate(X_train,Y_train)
_,accuracy_validation_drop = model_dropout_part3.evaluate(X_validation,Y_validation)
_,accuracy_test_drop = model_dropout_part3.evaluate(X_test,Y_test)

print('Accuracy of Training Dataset Classified with Regularization in Neural Network is: %.2f' % (accuracy_train_reg*100))
print('Accuracy of Training Dataset Classified with Dropout in Neural Network is: %.2f' % (accuracy_train_drop*100))

print('Accuracy of Validation Dataset Classified with Regularization in Neural Network is: %.2f' % (accuracy_validation_reg*100))
print('Accuracy of Validation Dataset Classified with Dropout in Neural Network is: %.2f' % (accuracy_validation_drop*100))

print('Accuracy of Test Dataset Classified with Regularization in Neural Network is: %.2f' % (accuracy_test_reg*100))
print('Accuracy of Test Dataset Classified with Dropout in Neural Network is: %.2f' % (accuracy_test_drop*100))




