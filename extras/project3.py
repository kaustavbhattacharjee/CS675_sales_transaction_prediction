import numpy as np
from sklearn.linear_model import Ridge


class LstmMethod():

    def __init__(self,train_data,window_size):
        self.window_size=window_size
        self.train_data=train_data
        self.data_size=len(train_data)
        self.new_train=[]
        self.new_labels=[]
        self.test_data=[[]]
        for i in range(self.data_size-self.window_size):
            x=[]
            for j in range(self.window_size):
                x.append(self.train_data[i+j])
            self.new_train.append(x)
            self.new_labels.append(self.train_data[i+self.window_size])
        for i in range(self.window_size):
            self.test_data[0].append(self.train_data[self.data_size-self.window_size+i])

    def ridgeRegression(self):
        clf = Ridge()
        clf.fit(self.new_train, self.new_labels)
        pred = clf.predict(self.test_data)
        return pred


def mean_squared_error(prediction,test):
    sum=0
    n=len(prediction)
    for i in range(n):
        sum+=((prediction[i]-test[i])**2)
    return sum/n

data = np.genfromtxt('Sales_Transactions_Dataset_Weekly.csv', delimiter=',')
train_data = data[1:, 1:52]
test_data=data[1:, 52:53] # taking the last week data as test data
test_data=[item[-1] for item in test_data]
window=51
product_num,week_num=np.shape(train_data)
#print('Dimension of train data: {} X {}'.format(product_num,week_num))

min_error=float('inf')
min_w=0
min_error_pred_data=[]
for w in range(2,window):
    pred_data=[]
    for i in range(product_num):
        object=LstmMethod(train_data[i],w)
        pred_data.append(object.ridgeRegression())
    error=mean_squared_error(pred_data,test_data)
    #print('Mean Square Error:{} for window:{}'.format(error,w))
    if error<min_error:
        min_error=error
        min_error_pred_data=pred_data
        min_w=w

for i in range(len(min_error_pred_data)):
    print('{}\t{}'.format(round(min_error_pred_data[i][0],1),(i+1)))
print('Minimum Mean Square Error:{}'.format(round(min_error[0],2),min_w))
