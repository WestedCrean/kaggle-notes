import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=64)

# Average CV score on the training set was: -0.447279695265272
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.15000000000000002, min_samples_leaf=4, min_samples_split=5, n_estimators=100, subsample=0.9500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 64)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
