import numpy as np

def my_cl_acc(pred, gt):
    how_many_correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            how_many_correct += 1

    return 100 * (how_many_correct / len(pred))


def my_1nn(x_train, y_train, x_test):
    y_test = []
    for index_of_x_test in range(len(x_test)):
        best_index = 0
        dist_best = float('inf')
        for index_of_x_train in range(len(x_train)):

            dist = np.linalg.norm(x_test[index_of_x_test] - x_train[index_of_x_train])
            if dist_best > dist:
                dist_best = dist
                best_index = index_of_x_train
        y_test.append(y_train[best_index])
    return np.array(y_test)
