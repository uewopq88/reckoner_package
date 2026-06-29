import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def identf_split(X_train, y_train, c_threshold=0.6):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print("Identification finished")

    confidences = clf.predict_proba(X_train)
    confidences = confidences.max(axis=1)
    confidences = pd.DataFrame(confidences, columns=['confidnece_score'])
    confidences['index'] = list(range(0,len(confidences)))

    confidences['identification'] = np.where(
        confidences['confidnece_score'] >= c_threshold, True, False)

    df_high = confidences[confidences['identification'] == True]
    df_low = confidences[confidences['identification'] != True]

    high_conf_ind = df_high['index'].values
    low_conf_ind = df_low['index'].values

    return high_conf_ind, low_conf_ind
