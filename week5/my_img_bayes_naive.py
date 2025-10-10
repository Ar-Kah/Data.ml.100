import numpy as np
import scipy.stats import multivariate_normal
import pickle
from tqdm import tqdm


def my_calc_meanvar(x_train, y_train):
    """
    function to calculate the mean and variance vectors
    for 10 classes
    """

    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_train_flat = x_train_flat + np.random.normal(loc=0.0, scale=10, size=x_train_flat.shape)
    class_bins = {k: [] for k in range(10)}
    for index, image_vector in enumerate(x_train_flat):

        binn = y_train[index]
        class_bins[binn].append(image_vector)

    class_bins = {k: np.array(v) for k, v in class_bins.items()}

    # Count means for every bin
    means = {k: x_train_flat[y_train == k].mean(axis=0) for k in range(10)}
    variances = {k: x_train_flat[y_train == k].var(axis=0)  for k in range(10)}
    return means, variances

def my_cl_acc(pred, gt):
    """Vectorized accuracy calculation."""
    pred = np.ravel(pred)
    gt = np.ravel(gt)
    return 100 * np.mean(pred == gt)


def main():
    # Load Fashion-MNIST data
    with open('mnist_fashion.pkl', 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    mean_l, var_l = my_calc_meanvar(x_train, y_train)

    # Compute priors
    priors = np.array([np.mean(y_train == k) for k in range(10)])

    # Compute log-likelihoods
    log_p_class = []
    for k in range(10):
        mean_k = mean_l[k]       # (D,)
        var_k = var_l[k] + 1e-6 # avoid division by zero
        term = np.log(2 * np.pi) + np.log(var_k) + ((x_test_flat - mean_k) ** 2) / var_k
        log_likelihood = -0.5 * np.sum(term, axis=1)       # (M,)
        log_prior = np.log(priors[k])                      # scalar
        log_p_class.append(log_likelihood + log_prior)

    log_p_class = np.vstack(log_p_class).T  # shape (M, 10)
    pred = np.argmax(log_p_class, axis=1)   # (M,)
    acc = my_cl_acc(pred, y_test)
    print(f"my acc: {acc}")
    # Save predictions
    np.savetxt('PRED_bayes.dat', pred, fmt='%d')

if __name__ == '__main__':
    main()
