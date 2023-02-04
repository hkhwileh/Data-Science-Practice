import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\\Z\\ex2data1.txt'

data = pd.read_csv(path,header=None,
                   names=['Exam1','Exam2','Admitted'])
print('data = ')
print(data.head(10))
print('==========================')
print('data.discribe = ')
print(data.describe())

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]


#print('Admitted student \n',positive)
#print('==========================')
#print('NotAdmitted student \n',negative)
print('==========================')
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['Exam1'], positive['Exam2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'],s=50,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


def sigmoid(z):
    return 1/(1+np.exp(-z))

nums = np.arange(-10,10,step=1)
fig,ax =plt.subplots(figsize=(8,5))

ax.plot(nums,sigmoid(nums),'r')
plt.show()
print('==========================')
data.insert(0,'ones',1)
#print('new data ',data)
#print('==========================')

cols = data.shape[1]
X= data.iloc[:,0:cols-1]
Y=data.iloc[:,cols-1:cols]

print('X \n',X)
print('==========================')
print('Y \n',Y)

X = np.array(X.values)
Y = np.array(Y.values)

theta = np.zeros(3)
print('X \n',X)
print('==========================')
print('Y \n',Y)


def cost(thetav,Xv,Yv):
    thetav = np.matrix(thetav)
    xv = np.matrix(Xv)
    yv = np.matrix(Yv)
    first = np.multiply(-yv,np.log(sigmoid(xv*thetav.T)))
    second = np.multiply((1-yv),np.log(1-sigmoid(xv*thetav.T)))
    return np.sum(first-second)/(len(xv))

thiscost = cost(theta,X,Y)
print('==========================')
print('Cost = ', thiscost)

def gradient(thetav,Xv,Yv):
    thetav = np.matrix(thetav)
    Xv = np.matrix(Xv)
    yv = np.matrix(Yv)

    parameters = int(thetav.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(Xv* thetav.T) - yv

    for i in range(parameters):
        term = np.multiply(error,Xv[:,i])
        grad[i]= np.sum(term) /len(Xv)
    return grad

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,Y))
print('==========================')
print('result = ', result)


costafteroptimize = cost(result[0],X,Y)

print('==========================')
print('cost after optimze = ', costafteroptimize)


def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predections = predict(theta_min,X)

print('==========================')
print('new predict =  ', predections)

correct = [1 if ((a==1 and b==1) or
                 (a==0 and b==0)) else 0
           for (a,b) in zip(predections,Y)]

accuracy = ((sum(map(int,correct)) % len(correct)))

print('accuracy = ', accuracy)