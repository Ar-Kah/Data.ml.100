import pickle
from utils import my_1nn, my_cl_acc

def main():
    data_fname = 'mnist_fashion.pkl'

    with open(data_fname, 'rb') as data_file:
        x_train = pickle.load(data_file)
        y_train = pickle.load(data_file)
        x_test = pickle.load(data_file)
        y_test = pickle.load(data_file)


    # print the size of training and test data
    # print(f'x_train shape {x_train.shape}')
    # print(f'y_train shape {y_train.shape}')
    # print(f'x_test shape {x_test.shape}')
    # print(f'y_test shape {y_test.shape}')

    pred = my_1nn(x_train, y_train, x_test)
    print(f"My prediction accuracy: {my_cl_acc(pred, y_test)}")

if __name__ == '__main__':
    main()
