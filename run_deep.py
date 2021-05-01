import datetime
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model import BinaryClassification
from src.utils.utils import binary_acc, area_under_the_curve, get_shap_values, create_csv_submission
from src.utils.get_data import import_data, get_data_loader, split_experts
from src.utils.config import SEED
from src.utils.model_helpers import weight_reset
from src.utils.preprocessing import preprocessing_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = './data'
GS_DIR = "./models/grid_search_results"
PARAM_DIR = "./models/weights"
PREDICTION_DATA = "./data/test"


def train_model(X_train,
                y_train,
                subject_indices,
                model="binary",
                criterion="BCE",
                optimizer="SGD",
                smote=False,
                activation_function="relu",
                batch_size=128,
                hidden_layer_dims=[150],
                epochs=1000,
                learning_rate=0.0001,
                dropout=0.5,
                weight_decay=0.5,
                split_val=0.33,
                verbose=True,
                random_state=SEED):
    """
    Modular Function for initializing and training a model.

    Args:
        X_train (np.array): training samples
        y_train (np.array): training labels
        subject_indices (np.array): indices of distinct subjects
        model (string): type of model; currently implemented: ["binary"]
        criterion (string): loss function; currently implemented: ["BCE"]
        optimizer (string): optimizer; currently implemented: ["Adam"]
        smote (bool): whether to perform SMOTE class equalization
        activation_function (string): which activation function; currently
                                            implemented: ["relu"]
        batch_size (int): size of training batches
        hidden_layer_dims (list of ints): the dimensions (and amount) of hidden
                                            layer in the neural network. Ordered
                                            from input to output (without input
                                            and output layer sizes)
        epochs (int): max number of training epochs
        learning_rate (float): learning rate for the model
        dropout (float): dropout rate
        weight_decay (float): weight decay rate (i.e. regularization)
        split_val (float): validation dataset size
        verbose (bool): print information
        random_state (int): seed for random functions
    Returns:
        model (nn.Module): A trained model
    """

    # split them into train and test according to the groups
    gss = GroupShuffleSplit(n_splits=1, train_size=1 - split_val, random_state=SEED)
    # since we only split once, use this command to get the
    # corresponding train and test indices
    for train_idx, val_idx in gss.split(X_train, y_train, subject_indices):
        continue

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    # apply class imbalance equalization using SMOTE
    if smote:
        oversample = SMOTE(random_state=random_state)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        y_train = y_train.reshape((-1, 1))

    # create torch data loaders for training and validation
    train_loader = get_data_loader(X_train, y_train, batch_size)
    val_loader = get_data_loader(X_val, y_val, X_val.shape[0])

    # initiate the correct model

    assert model in "binary", "Model not implemented yet!"

    if model == "binary":
        model = BinaryClassification([X_train.shape[1], *hidden_layer_dims, 1],
                                     dropout=dropout,
                                     activation_function=activation_function)
    # cpu or gpu, depending on setup
    model.to(device)

    # print model summary
    if verbose:
        print(model)

    # initiate the correct criterion/loss function
    if criterion == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    # initiate the correct optimizer
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, momentum=0.5)

    # saves validation losses over all epochs
    val_losses_epochs = []
    val_acc_epochs = []
    val_auc_epochs = []

    # used to store weights of the best model
    best_model = None
    # indicates best index of the epoch that produced the best model
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        # set pytorch model into training mode (relevant for batch norm, dropout, ..)
        model.train()

        # go through all batches
        for X_batch, y_batch in train_loader:
            # map them to cpu/gpu tensors
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # set up optimizer
            optimizer.zero_grad()

            # make a prediction with the model
            y_pred = model(X_batch)

            # taking care of dims
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 0]
            if y_batch.ndim == 2:
                y_batch = y_batch[:, 0]

            # calculate the loss of the prediction
            loss = criterion(y_pred, y_batch)

            # perform backpropagation
            loss.backward()
            # use the optimizer to update the weights for this batch
            optimizer.step()

        # put model into evaluation mode
        model.eval()

        # validation run
        for X_batch, y_batch in val_loader:
            # make a prediction
            y_pred = model(X_batch)

            # taking care of dims
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 0]
            if y_batch.ndim == 2:
                y_batch = y_batch[:, 0]
            # print(y_pred)
            # print(y_batch)
            loss = criterion(y_pred, y_batch)

            # calculate acc and auc of the prediction
            acc = binary_acc(y_pred, y_batch)

            # calculate the auc, while taking into consideration, that
            # a batch might not contain positive samples
            if len(np.unique(y_batch)) == 1:
                auc = "None"
            else:
                auc = area_under_the_curve(y_pred.detach().numpy(), y_batch.detach().numpy())

        # add the average loss, acc and auc to lists
        val_losses_epochs.append(np.mean(loss.item()))
        val_acc_epochs.append(acc)
        val_auc_epochs.append(auc)

        # for printing
        indicator_string = ""

        # update the best model, if the current epoch produced lowest auc/loss
        # use auc, if we never had a case that a batch contained only 0s
        try_acc = False
        if not ("None" in val_auc_epochs) and (np.max(val_auc_epochs) == val_auc_epochs[-1]):
            best_epoch = epoch
            best_model = model.state_dict()
            indicator_string += "!"
        elif "None" in val_auc_epochs:
            try_acc = True
        # otherwise use loss
        if try_acc and np.min(val_losses_epochs) == val_losses_epochs[-1]:
            best_epoch = epoch
            best_model = model.state_dict()
            indicator_string += "!"

        if verbose:
            if isinstance(auc, str):
                print(
                    f'Epoch {epoch + 0:03}: | Validation Loss: {val_losses_epochs[-1]:.3f}  | ACC: {val_acc_epochs[-1]:.3f} | AUC: {val_auc_epochs[-1]} {indicator_string}')
            else:
                print(
                    f'Epoch {epoch + 0:03}: | Validation Loss: {val_losses_epochs[-1]:.3f}  | ACC: {val_acc_epochs[-1]:.3f} | AUC: {val_auc_epochs[-1]:.3f} {indicator_string}')

        # convergence criterion: at least found a new best model in the last 5 epochs
        if (epoch - best_epoch) >= 5:
            break

    # load the best model from memory
    model.load_state_dict(best_model)

    return model


################################################################################
################################################################################
################################################################################

def predict(X, model, batch_size=1):
    """
    Given a trained model and a prediction dataset, this function will produce
    predictions.

    Args:
        X (np.array): prediction samples
        model (nn.Module): a trained model
        batch_size (int): batch size for testing (usually 1)

    Returns:
        (list): predictions
    """

    # create a torch data loader for training
    test_loader = get_data_loader(X, [0.] * X.shape[0], batch_size)

    # set model into testing mode
    model.eval()

    # for saving predictions and the real labels
    y_pred_list = []

    with torch.no_grad():
        # go through all data points
        for X_batch, y_batch in test_loader:
            # map to cpu/gpu tensors
            X_batch = X_batch.to(device)
            # get predictions
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)

            y_pred_list.extend(y_test_pred.cpu().numpy())

    return y_pred_list


################################################################################
################################################################################
################################################################################

def test_model(X_test, y_test, model, batch_size=1, verbose=True):
    """
    Given a trained model and a test dataset, this function will measure the
    performance of the model.

    Args:
        X_test (np.array): testing samples
        y_test (np.array): testing labels
        model (nn.Module): a trained model
        batch_size (int): batch size for testing (usually 1)
        verbose (bool): printing extra information
    Return:
        acc (float): accuracy of the model on the testing dataset
        auc (float): area under the curve of the model on the testing dataset
    """

    # create a torch data loader for training
    test_loader = get_data_loader(X_test, y_test, batch_size)

    # set model into testing mode
    model.eval()

    # for saving predictions and the real labels
    y_pred_list = []
    y_pred_tag_list = []
    y_test_list = []

    with torch.no_grad():
        # go through all data points
        for X_batch, y_batch in test_loader:
            # map to cpu/gpu tensors
            X_batch = X_batch.to(device)
            # get predictions
            y_test_pred = model(X_batch)
            # since this is training, use sigmoid to map it between 0 and 1
            # print("\n"*10)
            # print(y_test_pred)
            y_test_pred = torch.sigmoid(y_test_pred)
            # print(y_test_pred)
            # ... and round it to 0 or 1
            y_pred_tag = torch.round(y_test_pred)
            # add to list
            y_pred_list.extend(y_test_pred.cpu().numpy())
            y_pred_tag_list.extend(y_pred_tag.cpu().numpy())
            y_test_list.extend(y_batch.cpu().numpy())
            # print(y_pred_tag)
    if verbose:
        print("[!] Confusion Matrix")
        print(confusion_matrix(y_test_list, y_pred_tag_list))
        print("[!] Classification Report")
        print(classification_report(y_test_list, y_pred_tag_list))

    # calculate testing accuracy and area under the curve
    auc = area_under_the_curve(y_pred_list, y_test_list)
    acc = binary_acc(y_pred_tag_list, y_test_list)

    if verbose:
        print(f"Test finished with ACC: {acc} | AUC: {auc}")

    return acc, auc


################################################################################
################################################################################
################################################################################
def cross_validation_nn(X,
                        y,
                        subjects,
                        K,
                        segmentation_type,
                        file_name=None,
                        train_size=0.7,
                        models=["binary"],
                        hidden_layer_dims=[[50], [100], [200], [400]],
                        batch_sizes=[128],
                        learning_rates=[0.0005, 0.0001],
                        criteria=["BCE"],
                        optimizers=["Adam"],
                        activation_functions=["relu"],
                        weight_decays=[0.2, 0.5],
                        dropouts=[.5],
                        smote_setups=[True, False],
                        max_epochs=1000,
                        type_of_data="whole_data",
                        using_user_features=True,
                        random_state=42,
                        verbose=False):
    """
    Function to perform cross validation and find the best hyperparameters.

    Args:
        X (np.array): training samples
        y (np.array): training labels
        subjects (list): indexes for each sample, representing the individual of
                            who it is from. used in order to not use one
                            individuals samples in test and train
        K (int): number of splits
        segmentation_type (str): no/coarse/fine segmentation (for logging)
        file_name (str): if given, used as the file name, otherwise it is
                            created automatically
        hidden_layer_dims (list): list of hidden layer setups
        batch_sizes (list): list of batch sizes
        learning_rates (list): list of learning rates
        criteria (list): list of criteria/loss functions
        optimizers (list): list of optimizers
        activation_functions (list): list of activation functions
        weight_decays (list): list of weight decays
        dropouts (list): list of dropouts
        smote_setups (list): list of smote setups
        max_epochs (int): number of max epochs per training cycle
        type_of_data (str): either whole data, or for a specific expert (for
                                        logging)
        using_user_features (bool): whether this data contains user data or not
                                        (for logging)
        random_state (int): setting fixed seeds
        verbose (bool): printing extra information
    """

    # get a string with the current datetime, for naming the produced grid
    # search file
    start_time = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    # dictionary for saving the results
    gs_dic = {"model": [],
              "criterion": [],
              "optimizer": [],
              "activation_function": [],
              "hidden_layer_dims": [],
              "batch_size": [],
              "learning_rate": [],
              "weight_decay": [],
              "dropout": [],
              "smote": [],
              "avg_acc": [],
              "avg_auc": [],
              "use_user_data": [],
              "split_by_expert": [],
              "segmentation_type": []}

    # this functions enables us to split into multiple training/test dataset,
    # i.e. cross validation set, while also upholding the condition to not mix
    # samples from the same individual/group into both training and testing at
    # the same time
    gss = GroupShuffleSplit(n_splits=K, train_size=1 - 1 / K, random_state=random_state)

    # bring smote in
    for model in models:
        for criterion in criteria:
            for optimizer in optimizers:
                for activation_function in activation_functions:
                    for dims in hidden_layer_dims:
                        for batch_size in batch_sizes:
                            for learning_rate in learning_rates:
                                for weight_decay in weight_decays:
                                    for dropout in dropouts:
                                        for smote in smote_setups:
                                            # measure for evaluating each setup
                                            sum_acc = 0
                                            sum_auc = 0
                                            print(
                                                f"[!] SETUP: {model} | {criterion} | {optimizer} | {activation_function} | {dims} | {learning_rate} | {weight_decay} | {dropout} | {smote}")
                                            # perform cross validation
                                            for train_idx, test_idx in gss.split(X, y, subjects):
                                                # train a model with the given hyperparameters,
                                                # and the given cv split dataset
                                                cv_model = train_model(X[train_idx],
                                                                       y[train_idx],
                                                                       [subjects[x] for x in train_idx],
                                                                       model=model,
                                                                       criterion=criterion,
                                                                       optimizer=optimizer,
                                                                       activation_function=activation_function,
                                                                       hidden_layer_dims=dims,
                                                                       batch_size=batch_size,
                                                                       learning_rate=learning_rate,
                                                                       weight_decay=weight_decay,
                                                                       dropout=dropout,
                                                                       epochs=max_epochs,
                                                                       smote=smote,
                                                                       verbose=verbose)
                                                # test the cv model, and add the
                                                # performance of this cv split
                                                # to the measures
                                                acc, auc = test_model(X[test_idx],
                                                                      y[test_idx],
                                                                      cv_model,
                                                                      verbose=False)
                                                sum_acc += acc
                                                sum_auc += auc

                                                # each time, reset the model weights
                                                cv_model.apply(weight_reset)
                                            # print the average performance
                                            print(f"ACC {sum_acc / K:.3f} | AUC {sum_auc / K:.3f}")
                                            # save hyperparameters
                                            gs_dic["model"].append(model)
                                            gs_dic["criterion"].append(criterion)
                                            gs_dic["optimizer"].append(optimizer)
                                            gs_dic["activation_function"].append(activation_function)
                                            gs_dic["hidden_layer_dims"].append(dims)
                                            gs_dic["batch_size"].append(batch_size)
                                            gs_dic["learning_rate"].append(learning_rate)
                                            gs_dic["weight_decay"].append(weight_decay)
                                            gs_dic["dropout"].append(dropout)
                                            gs_dic["smote"].append(smote)
                                            gs_dic["avg_acc"].append(sum_acc / K)
                                            gs_dic["avg_auc"].append(sum_auc / K)
                                            gs_dic["use_user_data"].append(using_user_features)
                                            gs_dic["split_by_expert"].append(type_of_data)
                                            gs_dic["segmentation_type"].append(segmentation_type)
                                            # transform to dictionary
                                            gs_df = pd.DataFrame.from_dict(gs_dic)
                                            # save dictionary
                                            gs_df.to_pickle(GS_DIR + "/" + segmentation_type + "_gs.pkl")


################################################################################
################################################################################
################################################################################


def predict_test_results():
    """
    Function that created predictions, given that models are in
    models/weights. Saves them into data/test/predictions_deep.
    """

    segmentation_types = ["no", "coarse", "fine"]
    model_paths = ["no_model.pth", "coarse_model.pth", "fine_model.pth"]

    # create folder if not already there
    Path(PREDICTION_DATA + "/predictions_deep").mkdir(exist_ok=True)

    for segmentation, model_path in zip(segmentation_types, model_paths):
        # load the best model
        model = torch.load(PARAM_DIR + "/" + model_path)
        model.eval()

        # get the corresponding data
        X = import_data(DATA_PATH, segmentation_type=segmentation,
                        drop_user_features=False,
                        drop_expert=True,
                        is_test=True)

        # preprocess it
        X = preprocessing_pipeline(X, drop_corr=False)

        # make predictions
        predictions = predict(X.values, model)
        predictions = [x[0] for x in predictions]

        # save predictions
        create_csv_submission(predictions, segm_type=segmentation,
                              submission_path=PREDICTION_DATA + "/predictions_deep",
                              expert=False, user_features=True)


def train_best_models():
    """
    Trains the best models using grid search results in "models/grid_search_results"
    and saves them in "models/weights"
    """
    segmentation_types = ["no", "coarse", "fine"]
    gs_paths = ["no_gs.pkl", "coarse_gs.pkl", "fine_gs.pkl"]
    for segmentation, path in zip(segmentation_types, gs_paths):
        # get feature and label dataframes
        features, labels = import_data(path=DATA_PATH,
                                       segmentation_type=segmentation,
                                       drop_user_features=False,
                                       return_type='pd',
                                       drop_expert=True)
        # get the best parameters
        df = pd.read_pickle(GS_DIR + "/" + path)
        best_params = df.iloc[df['avg_auc'].argmax()]

        print(f"Generating best models for -->{segmentation}<-- data!")

        # get the the subject indices, in order to avoid putting samples from
        # the same individuals into test and training
        subject_indices = [l[0] for l in list(features.index)]

        # preprocess our features
        features = preprocessing_pipeline(features, drop_corr=False)

        # train a model
        model = train_model(features.values,
                            labels.values,
                            subject_indices,
                            optimizer=best_params["optimizer"],
                            smote=best_params["smote"],
                            hidden_layer_dims=best_params["hidden_layer_dims"],
                            learning_rate=best_params["learning_rate"],
                            dropout=best_params["dropout"],
                            weight_decay=best_params["weight_decay"],
                            split_val=0.2,
                            verbose=True,
                            epochs=1000)

        # save this best model
        torch.save(model, PARAM_DIR + "/" + segmentation + "_model.pth")


def grid_search():
    segmentation_type = sys.argv[2]
    split_by_expert = sys.argv[3] == "True"
    drop_user_features = sys.argv[4] == "True"

    # get feature and label dataframes
    features_whole, labels_whole = import_data(path=DATA_PATH,
                                               segmentation_type=segmentation_type,
                                               drop_user_features=drop_user_features,
                                               return_type='pd',
                                               drop_expert=not split_by_expert)
    # if we dont split by expert, we only have one pair or features/labels
    if not split_by_expert:
        data = {"whole_data": (features_whole, labels_whole)}
    # otherwise we split them into the three experts
    else:
        temp_data = split_experts(features_whole, labels_whole)
        data = {f"expert_{int((i / 2) + 1)}": (temp_data[i], temp_data[i + 1]) for i in range(0, len(temp_data), 2)}

    # go through the dataframes (only one if we dont split)
    for name, (features, labels) in data.items():
        print(f"Looking at -->{name}<-- data!")

        # get the the subject indices, in order to avoid putting samples from
        # the same individuals into test and training
        subject_indices = [l[0] for l in list(features.index)]

        # preprocess our features
        features = preprocessing_pipeline(features, drop_corr=False)

        # cross validation
        cross_validation_nn(features.values, labels.values, subject_indices,
                            K=3, verbose=False,
                            segmentation_type=segmentation_type,
                            using_user_features=not drop_user_features,
                            type_of_data=name)


def train_test():
    # if no command line arguments are given, use predefined ones
    if len(sys.argv) == 1:
        # to split the coughs by expert or not
        split_by_expert = False
        # to use the user supplied data
        drop_user_features = False
        # no, coarse or fine segmentation data
        segmentation_type = "coarse"
    # else, take the command line arguments
    else:
        segmentation_type = sys.argv[2]
        split_by_expert = sys.argv[3] == "True"
        drop_user_features = sys.argv[4] == "True"

    # get feature and label dataframes
    features_whole, labels_whole = import_data(path=DATA_PATH,
                                               segmentation_type=segmentation_type,
                                               drop_user_features=drop_user_features,
                                               return_type='pd',
                                               drop_expert=not split_by_expert)
    # if we dont split by expert, we only have one pair or features/labels
    if not split_by_expert:
        data = {"whole_data": (features_whole, labels_whole)}
    # otherwise we split them into the three experts
    else:
        temp_data = split_experts(features_whole, labels_whole)
        data = {f"expert_{int((i / 2) + 1)}": (temp_data[i], temp_data[i + 1]) for i in range(0, len(temp_data), 2)}

    # go through the dataframes (only one if we dont split)
    for name, (features, labels) in data.items():
        print(f"Looking at -->{name}<-- data!")

        # get the the subject indices, in order to avoid putting samples from
        # the same individuals into test and training
        subject_indices = [l[0] for l in list(features.index)]

        # preprocess our features
        features = preprocessing_pipeline(features, drop_corr=False)

        # split them into train and test according to the groups
        gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
        # since we only split once, use this command to get the
        # corresponding train and test indices
        for train_idx, test_idx in gss.split(features.values, labels.values, subject_indices):
            continue

        # train a model
        model = train_model(features.values[train_idx],
                            labels.values[train_idx],
                            [subject_indices[x] for x in train_idx],
                            verbose=True,
                            epochs=300)
        # calculate the SHAP values
        shap_df = get_shap_values(model,
                                  features.values[train_idx],
                                  features.values[test_idx],
                                  features.columns,
                                  device=device)
        print("\n\n\n SHAP Values")
        print(shap_df)

        # test the model
        test_model(features.values[test_idx], labels.values[test_idx], model, verbose=True)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 1:
        train_test()
    elif len(sys.argv) == 2 and sys.argv[1] == "make_predictions":
        predict_test_results()
    elif len(sys.argv) == 2 and sys.argv[1] == "train_best_models":
        train_best_models()
    elif len(sys.argv) == 5 and sys.argv[1] == "train_test":
        train_test()
    elif len(sys.argv) == 5 and sys.argv[1] == "grid_search":
        grid_search()
    else:
        print("Not possible, check README for specific execution instructions!")
