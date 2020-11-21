import numpy as np

test_data = np.loadtxt("mnist_test.csv",
                       delimiter=",")
test_len = 10000
random_subset_test = np.random.randint(test_len, size=100)
test_data = test_data[random_subset_test, :]
test_len = 100

label = 8
test_data[:, 1:] = test_data[:, 1:] / 255.0
# testing
wb = np.loadtxt('weights' + '_' + str(label) + '.csv', delimiter=',')
w = wb[:-1]
b = wb[-1]
correct_predictions = 0
for x in test_data:
    if np.dot(x[1:], w) + b > 0:
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
print(f"accuracy is {accuracy*100} percentage")
