# Since there are 60,000 points in the training set
# vector a = [a1,a2,...a60000] we have that many aphas in eqn (a)
# first we will initialize that vector randomly

import numpy as np

labels = 10 # 0,1,...,9

train_data = np.loadtxt("mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv",
                       delimiter=",")
train_len = 60000
test_len = 10000
# test_data.shape = (60000,785) 60,000 rows
# train_data.shape = (10000,785) //ly
# the actual size of image is 28*28 = 784
# the first character in every row is the label 

# I use gradient descent to solve the eqn (f)
# So the dot product is used in every step

#dot_products = dict()
#
#for i in range(train_len):
#    for j in range(i,train_len):
#        # one might want to rescale the pixel values of images in order to avoid overflow
#        res = np.dot(train_data[i,1:],train_data[j,1:])
#        # since dot product is commutative
#        dot_products[(i,j)],dot_products[(j,i)] = (res,res)

#def optimum(a,y):
#    complex_expression = 0
#    for i in range(train_len):
#        for j in range(train_len):
#            complex_expression += a[i] * a[j] * y[i] * y[j] * dot_products[(i,j)]
#
#    return sum(a) - 0.5 * complex_expression

def gradient(a,y):
    grad = np.zeros(train_len)
    for i in range(train_len):
        expr = 0
        for j in range(train_len):
            xi = train_data[i,1:]
            xj = train_data[j,1:]
            expr += a[i] * y[i] * y[j] * np.dot(xi,xj)
        grad[i] = a[i] - expr
    
    return grad

def gradient_descent(a,y):
    step_size = 0.001 # feel free to change
    # subtraction because we want to find the minimum
    return a - (step_size * gradient(a,y))

def classifier(label,epochs = 1000):
    # create the appropriate y vector
    y = np.ones(train_len)
    for i in range(train_len):
        if train_data[i][0] != label:
            y[i] = -1

    scale = 1.0 # feel free to change this number
    a = np.random.rand(train_len) * scale # randomly initialising alphas

    for i in range(epochs):
        a = gradient_descent(a,y)
    
    xs = train_data[:,1:] # ignoring the labels
    w = np.zeros(784) # same dimensions as a single point in data

    for i in range(train_len):
        w += (a[i]*y[i]) * xs[i]

    # L_min = optimum(a,y) error in training
    b = 0
    for i,v in enumerate(a):
        if v != 0:
            b = y[i] - np.dot(w,xs[i])
            break
    
    return w,b

label = 0
w,b = classifier(label) 
correct_predictions = 0

# testing

for x in test_data:
    if np.dot(x[1:],w) + b > 0:
        # positive prediction
        if label == x[0]:
            # true positive
            correct_predictions += 1
    else:
        # negative prediction
        if label != x[0]:
            # true negative
            correct_predictions += 1
accuracy = correct_predictions / test_len
print(f"accuracy is {accuracy}")
    
    



