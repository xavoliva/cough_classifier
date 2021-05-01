FEATURES = {
    'EEPD': ['EEPD50_100', 'EEPD100_150', 'EEPD150_200', 'EEPD200_250', 'EEPD250_300', 'EEPD300_350', 'EEPD350_400',
             'EEPD400_450', 'EEPD450_500', 'EEPD500_550', 'EEPD550_600', 'EEPD600_650', 'EEPD650_700', 'EEPD700_750',
             'EEPD750_800', 'EEPD800_850', 'EEPD850_900', 'EEPD900_950', 'EEPD950_1000'],
    'SPECTRAL': ['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Spread', 'Spectral_Skewness', 'Spectral_Kurtosis',
                 'Spectral_Bandwidth', 'Spectral_Flatness', 'Spectral_StDev', 'Spectral_Slope', 'Spectral_Decrease'],
    'PSD': ['PSD_225-425', 'PSD_450-550', 'PSD_1325-1600', 'PSD_1600-2000', 'PSD_2500-2900', 'PSD_3100-3700'],
    'PSD_FINE': ['PSD_225-425', 'PSD_450-550', 'PSD_1325-1600', 'PSD_1600-1900', 'PSD_2500-2900', 'PSD_3100-3700'],
    'MFCC': ['MFCC_mean0', 'MFCC_mean1', 'MFCC_mean2', 'MFCC_mean3', 'MFCC_mean4', 'MFCC_mean5', 'MFCC_mean6',
             'MFCC_mean7', 'MFCC_mean8', 'MFCC_mean9', 'MFCC_mean10', 'MFCC_mean11', 'MFCC_mean12', 'MFCC_std0',
             'MFCC_std1', 'MFCC_std2', 'MFCC_std3', 'MFCC_std4', 'MFCC_std5', 'MFCC_std6', 'MFCC_std7', 'MFCC_std8',
             'MFCC_std9', 'MFCC_std10', 'MFCC_std11', 'MFCC_std12'],
    'ADDITIONAL': ['Zero_Crossing_Rate', 'RMS_Power', 'Dominant_Freq', 'Crest_Factor', 'Cough_Length', 'SNR'],
    'METADATA': ['Age', 'Gender', 'Resp_Condition', 'Symptoms'],
}

ALL_FEATURES_COARSE = FEATURES['EEPD'] + ['Zero_Crossing_Rate', 'RMS_Power', 'Dominant_Freq'] + FEATURES['SPECTRAL'] + \
                      FEATURES['MFCC'] + ['Crest_Factor', 'Cough_Length'] + FEATURES['PSD'] + ['Expert'] + \
                      FEATURES['METADATA']

ALL_FEATURES_FINE = FEATURES['EEPD'] + ['Zero_Crossing_Rate', 'RMS_Power', 'Dominant_Freq'] + FEATURES['SPECTRAL'] + \
                      FEATURES['MFCC'] + ['Crest_Factor', 'Cough_Length'] + FEATURES['PSD_FINE'] + ['Expert'] + \
                      FEATURES['METADATA']

ALL_FEATURES_NO = FEATURES['EEPD'] + ['Zero_Crossing_Rate', 'RMS_Power', 'Dominant_Freq'] + FEATURES['SPECTRAL'] + \
                  FEATURES['MFCC'] + ['Crest_Factor', 'Cough_Length', 'SNR'] + FEATURES['PSD'] + ['Expert'] + \
                  FEATURES['METADATA']
CATEGORICAL_COLS = ['Expert', 'Gender', 'Resp_Condition', 'Symptoms']


SEED = 42

# GRID SEARCH PARAMETERS - STANDARD MODELS
KNN_PARAMS = {'n_neighbors': list(range(1, 16)), 'oversampling': [True, False]}
LOGISTIC_PARAMS = {'max_iter': [10000, 100000], 'oversampling': [True, False]}
LDA_PARAMS = {'oversampling': [True, False]}
SVC_PARAMS = {'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.01], 'oversampling': [True, False]}
NAIVE_BAYES_PARAMS = {'oversampling': [True, False]}
DECISION_TREE_PARAMS = {'max_depth': [3, 5, 7], 'oversampling': [True, False]}
RANDOM_FOREST_PARAMS = {'max_depth': [3, 5, 7], 'n_estimators': [3, 5, 7], 'oversampling': [True, False]}
GRADIENT_BOOSTING_PARAMS = {'max_depth': [3, 5, 7], 'n_estimators': [3, 5, 7], 'oversampling': [True, False]}
