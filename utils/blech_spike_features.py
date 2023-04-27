import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA

def get_dir_names():
    home_dir = os.getenv('HOME')
    blech_clust_dir = os.path.join(home_dir, 'Desktop', 'blech_clust')
    blech_dir_path = os.path.join(blech_clust_dir, 'blech.dir')
    with open(blech_dir_path, 'r') as f:
        lines = f.readlines()
    dir_name = lines[0][:-1]
    return home_dir, blech_clust_dir, dir_name

home_dir, blech_clust_dir, data_dir_name = get_dir_names()
sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata
metadata_handler = imp_metadata([[], data_dir_name])

params_dict = metadata_handler.params_dict
sampling_rate = params_dict['sampling_rate']
zero_ind = int(params_dict['spike_snapshot_before']*sampling_rate/1000)

#with open('./params/data_params.json', 'r') as path_file:
#    data_params = json.load(path_file)
#zero_ind = data_params['zero_ind']

class EnergyFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        energy = np.sqrt(np.sum(X**2, axis=-1))/X.shape[-1]
        energy = energy[:, np.newaxis]
        return energy

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class AmpFeature(BaseEstimator, TransformerMixin):
    def __init__(self, zero_ind):
        self.zero_ind = zero_ind

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        amplitude = X[:, self.zero_ind]
        amplitude = amplitude[:, np.newaxis]
        return amplitude

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def zscore_custom(x):
    return zscore(x, axis=-1)


zscore_transform = FunctionTransformer(zscore_custom)
log_transform = FunctionTransformer(np.log, validate=True)
arcsinh_transform = FunctionTransformer(np.arcsinh, validate=True)

pca_components = 3

pca_pipeline = Pipeline(
    steps=[
        ('zscore', zscore_transform),
        ('pca', PCA(n_components=pca_components)),
    ]
)

energy_pipeline = Pipeline(
    steps=[
        ('energy', EnergyFeature()),
        ('log', log_transform),
    ]
)

# use arcsinh_transform for amplitude due to negative values
amplitude_pipeline = Pipeline(
    steps=[
        ('amplitude', AmpFeature(zero_ind)),
        ('arcsinh', arcsinh_transform),
    ]
)

collect_feature_pipeline = FeatureUnion(
    n_jobs=1,
    transformer_list=[
        ('pca_features', pca_pipeline),
        ('energy', energy_pipeline),
        ('amplitude', amplitude_pipeline),
    ]
)

feature_pipeline = Pipeline(
    steps=[
        ('get_features', collect_feature_pipeline),
        ('scale_features', StandardScaler())
    ]
)

feature_names = ['pca_{}'.format(i) for i in range(pca_components)] + \
                ['energy', 'amplitude']
