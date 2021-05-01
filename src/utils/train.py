import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, explained_variance_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.utils.get_data import split_experts
from src.utils.preprocessing import oversample
from src.utils.model_helpers import cross_val_w_oversampling, ensemble_predictions
from collections import defaultdict

IMPLEMENTED_MODELS = (
    'knn', 'logistic', 'lda', 'svc', 'naive_bayes', 'decision_tree', 'random_forest', 'gradient_boosting')


def hyperparameter_tuning_cv(model, data, labels, cv_k, params,
                             metrics=(f1_score, roc_auc_score, accuracy_score,
                                      explained_variance_score)) -> pd.DataFrame:
    """
    Train the classical models hyperparameters with cross validation
    :param model: model to tune
    :type model: str
    :param data: data
    :type data: pd.DataFrame
    :param labels: true labels
    :type labels: pd.DataFrame
    :param cv_k: cv folds
    :type cv_k: int
    :param params: parameters to tune
    :type params: dictionary
    :param metrics: metrics for evaluation
    :type metrics: list
    :return: dataframe with evaluated parameters and metrics
    """

    assert model in IMPLEMENTED_MODELS, "Model does not exist"

    d = defaultdict(list)
    indexes_list = []
    for i in params.keys():
        indexes_list.append(i)
    param_grid = ParameterGrid(params)
    for param in param_grid:
        for key, value in param.items():
            d[key].append(value)
        scores_dict = None

        if model == 'knn':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=KNeighborsClassifier(n_neighbors=param["n_neighbors"]),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'logistic':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=LogisticRegression(max_iter=param["max_iter"]),
                                                   oversampling=param['oversampling'], metrics=metrics)

        if model == 'lda':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k, model=Lda(),
                                                   oversampling=param['oversampling'],
                                                   metrics=metrics)

        if model == 'svc':
            try:
                scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                       model=SVC(kernel=param['kernel'], gamma=param['gamma']),
                                                       oversampling=param['oversampling'], metrics=metrics)
            except ValueError:
                break
        if model == 'naive_bayes':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k, model=GaussianNB(),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'decision_tree':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=DecisionTreeClassifier(max_depth=param['max_depth']),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'random_forest':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=RandomForestClassifier(max_depth=param['max_depth'],
                                                                                n_estimators=param['n_estimators']),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'gradient_boosting':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                                                                    max_depth=param['max_depth']),
                                                   oversampling=param['oversampling'], metrics=metrics)

        for metric_name, score in scores_dict.items():
            d[metric_name].append(score)

    df = pd.DataFrame(data=d)
    df.set_index(indexes_list, inplace=True)
    return df


def train_predict(X_tr, y_tr, X_te, param):
    if param['oversampling']:
        X_tr, y_tr = oversample(X_tr, y_tr)

    clf = param['model'].fit(X_tr, y_tr.values.ravel())
    y_te_prob = pd.DataFrame(clf.predict_proba(X_te), columns=clf.classes_)

    return y_te_prob.iloc[:, 1]


def train_predict_experts(X_tr, y_tr, X_te, param):
    if param['oversampling']:
        X_tr, y_tr = oversample(X_tr, y_tr)

    X_tr_e1, y_tr_e1, X_tr_e2, y_tr_e2, X_tr_e3, y_tr_e3 = split_experts(X_tr, y_tr)

    fit_models = [
        param['models'][0].fit(X_tr_e1, y_tr_e1.values.ravel()),
        param['models'][1].fit(X_tr_e2, y_tr_e2.values.ravel()),
        param['models'][2].fit(X_tr_e3, y_tr_e3.values.ravel())
    ]

    y_te_prob = ensemble_predictions(fit_models, X_te, params=param["ensemble"])

    return y_te_prob[:, 1]
