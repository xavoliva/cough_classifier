# Define a function that automatically plots the AUC curve for a given classifier
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import plot_roc_curve, auc, roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from src.utils.preprocessing import oversample


def weight_reset(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train_test_split(X: pd.DataFrame, y: pd.DataFrame, random_state=42, fraction=0.7):
    """
    Split the data set in train and test data
    """
    X_tr = X.loc[X.sample(frac=fraction, random_state=random_state).index.unique()]  # .drop(['File_Name'], axis = 1)
    X_te = X.drop(X_tr.index)  # .drop(['File_Name'], axis = 1)

    y_tr = y.loc[y.sample(frac=fraction, random_state=random_state).index.unique()]  # .drop(['File_Name'], axis = 1)
    y_te = y.drop(y_tr.index)  # .drop(['File_Name'], axis = 1)
    y_te = y_te.Label
    y_tr = y_tr.Label

    return X_tr, y_tr, X_te, y_te


def cross_validation_iter(data: pd.DataFrame, labels: pd.DataFrame, k: int):
    """
    Compute cross validation for a single iteration.
    """
    unique_subjects = np.array(data.index.get_level_values('subject').unique())
    N = len(unique_subjects)
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    for k_iteration in range(k):
        k_indices = tuple([indices[k_iteration * fold_interval: (k_iteration + 1) * fold_interval]])
        # split the data accordingly into training and validation
        val_subjects = unique_subjects[k_indices]
        tr_mask = np.ones(N, bool)
        tr_mask[k_indices] = False
        tr_subjects = unique_subjects[tr_mask]

        x_tr = data.loc[tr_subjects]
        y_tr = labels.loc[tr_subjects]
        x_val = data.loc[val_subjects]
        y_val = labels.loc[val_subjects]

        yield x_tr, y_tr, x_val, y_val


def cross_validation(data, labels, k, model, metric):
    """
    perform k-fold cross-validation after dividing the training data into k folds
    """
    metric_list = []

    for x_tr, y_tr, x_val, y_val in cross_validation_iter(data, labels, k):
        k_fit = model.fit(x_tr, y_tr)
        y_pred = k_fit.predict(x_val)

        metric_list.append(metric(y_val, y_pred))

    return np.mean(metric_list)


def cross_val_w_oversampling(X, y, k, model, oversampling=True, metrics=[f1_score, roc_auc_score, accuracy_score]):
    """
    oversample the training data and perform k-fold cross-validation after dividing the training data into k folds, 
    grouped by subjects
    """
    groups = X.index
    gss = GroupShuffleSplit(n_splits=k, train_size=.7, random_state=42)
    gss.get_n_splits()

    metric_dict = defaultdict(list)
    return_dict = defaultdict(float)

    for train_idx, test_idx in gss.split(X, y, groups):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        if oversampling:
            X_tr, y_tr = oversample(X_tr, y_tr)

        k_fit = model.fit(X_tr, y_tr)
        y_pred = k_fit.predict(X_te)
        for metric in metrics:
            metric_dict[metric.__name__].append(metric(y_te, y_pred))

    for metric in metrics:
        return_dict[metric.__name__] = float(np.mean(metric_dict[metric.__name__]))
        if metric.__name__ == 'roc_auc_score':
            return_dict[metric.__name__ + '_var'] = float(np.var(metric_dict[metric.__name__]))

    return return_dict


def roc_w_cross_val(X, y, classifier, plot=False):
    cv = StratifiedKFold(n_splits=6)

    X = X.to_numpy()
    y = y.to_numpy()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, float(std_auc)),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    # ax.legend(loc="lower right")
    ax.legend(bbox_to_anchor=(1, 0), loc="lower left")

    if plot:
        plt.show()
    else:
        plt.close()

    return mean_auc


# make an ensemble prediction for multi-class classification
# source: https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
def ensemble_predictions(members, X_te, params):
    assert params["type"] in ("weighted", "stacked")
    # make predictions
    if params["type"] == "weighted":
        y_preds = np.array([model.predict_proba(X_te) for model in members])

        # mean across ensemble members
        y_ensemble_pred = np.average(y_preds, weights=params["weights"], axis=0)
    else:
        estimators = [(f'expert_{i}', members[i]) for i in range(len(members))]

        # only final estimator should be fitted here
        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression())
        X_tr = params["X_tr"]
        print(X_tr.columns.tolist())
        y_tr = params["y_tr"]

        clf.fit(X_tr, y_tr.values.ravel())

        y_ensemble_pred = clf.predict_proba(X_te)

    return y_ensemble_pred
