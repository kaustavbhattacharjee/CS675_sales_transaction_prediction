import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class LstmMethod():

    def __init__(self,training_data,size_of_window):
        self.size_of_window=size_of_window
        self.training_data=training_data
        self.size_of_data=len(training_data)
        self.new_training_data=[]
        self.new_labels=[]
        self.testing_data=[[]]
        for i in range(self.size_of_data-self.size_of_window):
            temp=[]
            for j in range(self.size_of_window):
                temp.append(self.training_data[i+j])
            self.new_training_data.append(temp)
            self.new_labels.append(self.training_data[i+self.size_of_window])
        for i in range(self.size_of_window):
            self.testing_data[0].append(self.training_data[self.size_of_data-self.size_of_window+i])

    def ridge_regression(self):
        classifier = Ridge()
        classifier.fit(self.new_training_data, self.new_labels)
        prediction = classifier.predict(self.testing_data)
        return prediction


def mse(prediction,test): # Mean Square Error
    sum=0
    n=len(prediction)
    for i in range(n):
        sum+=((prediction[i]-test[i])**2)
    return sum/n

datafile = np.genfromtxt('Sales_Transactions_Dataset_Weekly.csv', delimiter=',') #Reading the data file
datafile2 = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
product_codes = datafile2['Product_Code'].values
training_data = datafile[1:, 1:52] #Keeping Week 0 to 50's data for training
testing_data=datafile[1:, 52:53] #Keeping Week 51's data for testing
testing_data=[item[-1] for item in testing_data] #converting the weekly data into an array


number_of_products,week_number=np.shape(training_data)

#lstm and regression
minimum_error=float('inf')
minimum_w=0
minimum_error_predicted_data=[]
for w in range(2,51):
    pred_data=[]
    for i in range(number_of_products):
        object=LstmMethod(training_data[i],w) #creating the object for calling ridge regression
        pred_data.append(object.ridge_regression())
    error=mse(pred_data,testing_data)
    #updating the minimum error with the current error if it is the smallest
    if error<minimum_error:
        minimum_error=error
        minimum_error_predicted_data=pred_data
        minimum_w=w

for i in range(len(minimum_error_predicted_data)):
    print('{}\t{}'.format(product_codes[i],round(minimum_error_predicted_data[i][0],1)))
print('Minimum Mean Square Error:{}'.format(round(minimum_error[0],2))) #round the mse to 2 decimal points







