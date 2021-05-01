from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.utils.get_data import import_data
from src.utils.preprocessing import preprocessing_pipeline
from src.utils.train import train_predict, train_predict_experts
from src.utils.utils import create_csv_submission

# BEST MODEL PARAMETERS
BEST_PARAMS_NO_METADATA = {
    'coarse': {'model': SVC(kernel='linear', gamma=0.01, probability=True), 'oversampling': True},
    'fine': {'model': KNeighborsClassifier(n_neighbors=1), 'oversampling': True},
    'no': {'model': GaussianNB(), 'oversampling': True}
}

BEST_PARAMS_EXPERTS_NO_METADATA = {
    'coarse': {
        'models': [
            SVC(kernel='linear', gamma=0.1, probability=True),
            SVC(kernel='rbf', gamma=0.1, probability=True),
            SVC(kernel='rbf', gamma=0.1, probability=True)
        ],
        'oversampling': True,
    },
    'fine': {
        'models': [
            KNeighborsClassifier(n_neighbors=4),
            KNeighborsClassifier(n_neighbors=4),
            KNeighborsClassifier(n_neighbors=3)
        ],
        'oversampling': True,
    },
    'no': {
        'models': [
            SVC(kernel='linear', gamma=0.1, probability=True),
            SVC(kernel='rbf', gamma=0.01, probability=True),
            SVC(kernel='rbf', gamma=0.01, probability=True)
        ],
        'oversampling': True,
    },
}

BEST_PARAMS_WITH_METADATA = {
    'coarse': {'model': KNeighborsClassifier(n_neighbors=1), 'oversampling': True},
    'fine': {'model': KNeighborsClassifier(n_neighbors=1), 'oversampling': True},
    'no': {'model': GaussianNB(), 'oversampling': True}
}

BEST_PARAMS_EXPERTS_WITH_METADATA = {
    'coarse': {
        'models': [
            KNeighborsClassifier(n_neighbors=4),
            KNeighborsClassifier(n_neighbors=6),
            KNeighborsClassifier(n_neighbors=5)
        ],
        'oversampling': True,
    },
    'fine': {
        'models': [
            KNeighborsClassifier(n_neighbors=4),
            KNeighborsClassifier(n_neighbors=4),
            KNeighborsClassifier(n_neighbors=3)
        ],
        'oversampling': True,
    },
    'no': {
        'models': [
            GaussianNB(),
            GaussianNB(),
            GaussianNB()
        ],
        'oversampling': True,
    },
}

ENSEMBLE_TYPE = "weighted"

DATA_PATH = "./data"
SUBMISSION_PATH = "./data/test/predictions_classical"

if __name__ == "__main__":

    for segm_type, param in BEST_PARAMS_WITH_METADATA.items():
        X_tr, y_tr = import_data(DATA_PATH, segmentation_type=segm_type,
                                 drop_user_features=False,
                                 drop_expert=True)
        X_te = import_data(DATA_PATH, segmentation_type=segm_type,
                           drop_user_features=False,
                           drop_expert=True,
                           is_test=True)

        X_tr, X_te = preprocessing_pipeline(X_tr, X_te)

        y_pred = train_predict(X_tr, y_tr, X_te, param=param)
        create_csv_submission(y_pred, segm_type=segm_type, submission_path=SUBMISSION_PATH,
                              expert=False, user_features=True)

    #########################################################################

    for segm_type, param in BEST_PARAMS_NO_METADATA.items():
        X_tr, y_tr = import_data(DATA_PATH, segmentation_type=segm_type,
                                 drop_user_features=True,
                                 drop_expert=True)
        X_te = import_data(DATA_PATH, segmentation_type=segm_type,
                           drop_user_features=True,
                           drop_expert=True,
                           is_test=True)

        X_tr, X_te = preprocessing_pipeline(X_tr, X_te, dummy=False)

        y_pred = train_predict(X_tr, y_tr, X_te, param=param)
        create_csv_submission(y_pred, segm_type=segm_type, submission_path=SUBMISSION_PATH,
                              expert=False, user_features=False)

    #########################################################################

    for segm_type, param in BEST_PARAMS_EXPERTS_NO_METADATA.items():
        X_tr, y_tr = import_data(DATA_PATH, segmentation_type=segm_type,
                                 drop_user_features=True,
                                 drop_expert=False)
        X_te = import_data(DATA_PATH, segmentation_type=segm_type,
                           drop_user_features=True,
                           drop_expert=True,
                           is_test=True)

        expert_col = X_tr['Expert'].values
        X_tr.drop(['Expert'], axis=1, inplace=True)

        X_tr, X_te = preprocessing_pipeline(X_tr, X_te, stop=None, dummy=False)

        if ENSEMBLE_TYPE == "weighted":
            param["ensemble"] = {
                "type": "weighted",
                "weights": None,
            }
        else:
            param["ensemble"] = {
                "type": "stacked",
                "X_tr": X_tr,
                "y_tr": y_tr
            }

        X_tr['Expert'] = expert_col

        y_pred = train_predict_experts(X_tr, y_tr, X_te, param=param)
        create_csv_submission(y_pred, segm_type=segm_type, submission_path=SUBMISSION_PATH,
                              expert=True, user_features=False)
