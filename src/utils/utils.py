import torch
import numpy as np
import shap
import pandas as pd
import csv
from sklearn import metrics
from datetime import datetime


def binary_acc(y_pred, y_test):
    """
    Calculate accuracy of binary classification labels.

    Args:
        y_pred (list/np.array/torch.tensor): list of predictions
        y_test (list/np.array/torch.tensor): list of correct labels

    Returns:
        acc (float): Accuracy
    """
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().numpy()
    else:
        y_pred = np.array(y_pred)

    if torch.is_tensor(y_test):
        y_test = y_test.detach().numpy()
    else:
        y_test = np.array(y_test)

    y_pred_tag = np.array(np.round(torch.sigmoid(torch.tensor(y_pred))))
    correct_results_sum = np.sum([a == b for a, b in zip(y_pred_tag, y_test)])
    acc = correct_results_sum / y_test.shape[0]

    return acc


def area_under_the_curve(y_pred, y_test):
    """
    Calculate area under the curve of binary classification labels.

    Args:
        y_pred (list/np.array/torch.tensor): list of predictions
        y_test (list/np.array/torch.tensor): list of correct labels

    Returns:
        auc (float): Area under the curve
    """
    y_pred = y_pred
    y_test = y_test

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    return metrics.auc(fpr, tpr)


def get_shap_values(model, X_train, X_test, feature_names,
                    train_sample_size=1000, test_sample_size=300, device="cpu"):
    """
    Creates a dataframe of shap values for model, given train and testing parameters

    Args:
        model (torch.model): The trained pytorch model
        X_train (numpy.ndarray): Training Features
        X_test (numpy.ndarray): Testing features
        feature_names (list): Names of each feature for printing
        train_sample_sie (int): Size of training samples for SHAP calculation
        test_sample_sie (int): Size of testing samples for SHAP calculation
        device (string): cpu/gpu
    Returns:
        (pd.DataFrame): dataframe containing the shap values
    """

    train_sample_size = min(train_sample_size, X_train.shape[0])
    test_sample_size = min(test_sample_size, X_test.shape[0])
    # get training and testing samples
    train_samples = torch.from_numpy(X_train[np.random.choice(X_train.shape[0],
                                                              train_sample_size, replace=False)]).float().to(device)
    test_samples = torch.from_numpy(X_test[np.random.choice(X_test.shape[0],
                                                            test_sample_size, replace=False)]).float().to(device)

    # get the deep explainer
    de = shap.DeepExplainer(model, train_samples)
    # generate the shap values
    shap_values = de.shap_values(test_samples)

    # create a data frame with the absolute mean, std and names of the shap values
    shap_df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
        "feature_name": feature_names
    })

    # sort the entries by the mean shap value
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    return shap_df


def create_csv_submission(y_pred, segm_type, submission_path, expert, user_features):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """

    if user_features:
        metadata = "w_metadata"
    else:
        metadata = "no_metadata"

    if expert:
        metadata += "_expert_split"

    now = datetime.now()
    now_str = now.strftime("%H_%M_%S")
    name = f"predictions_{segm_type}_segmentation_{metadata}_{now_str}"
    with open(submission_path + "/" + name, 'w') as csv_file:
        fieldnames = ['Label']
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for row in y_pred:
            writer.writerow({'Label': row})
