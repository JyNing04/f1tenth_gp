"""	Train a GP model for error discrepancy between kinematic and dynamic models.
"""

# from trace import Trace
# import sklearn
# import torch
# import gpytorch
# import george 
# from george import kernels
import time
import numpy as np
import os
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, RationalQuadratic, DotProduct, Matern
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import sys
path = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/plots')
sys.path.insert(1, path)
from plots import plot_true_predicted_variance

#####################################################################
# load data

SAVE_MODELS = True
ORIGINAL    = False
FULL_DATA   = False
CTYPE       = 'PP'
VARIDX      = 6 # 5:ω, 6:β
map_index   = 0
map_list    = ['Sepang', 'Shanghai', 'YasMarina']
map_name    = map_list[map_index]
line_type   = 'raceline_ED' # raceline_ED or centerline
save_path   = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/gp_models/')
data_path   = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/')
downsample  = ['halfsample', 'downsample']
sample_idx  = 1 # 0:half or 1:one-third
data_size   = 'full' if FULL_DATA else downsample[sample_idx]
data_name   = ['f1tenth-DYN-{}-{}_{}', 'f1tenth-KIN-{}-{}_{}']
if not FULL_DATA:
    data_name = ['f1tenth-'+data_size+'-DYN-{}-{}_{}', 'f1tenth-'+data_size+'-KIN-{}-{}_{}']
if ORIGINAL:
    save_path += 'original/'
    data_path += 'original/'
elif line_type == 'centerline':
    save_path += 'centerline/'
    data_path += 'centerline/'
else:
    save_path += 'raceline/'
    data_path += 'raceline/'
state_names  = ['x', 'y', '𝛿', 'v', 'phi', 'ω', 'β']
Kernels_dict = {
    'RBF': 1.0*RBF(length_scale=1.0,length_scale_bounds=(1e-9, 1e9)),
    'PERIODIC': ExpSineSquared(length_scale=0.8, periodicity=2.0, length_scale_bounds=(1e-15, 100000.0), periodicity_bounds=(1e-15, 100000.0)),
    'RQ': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-09, 1e9), alpha_bounds=(1e-09, 1e9)),
    'LINEAR': DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-09, 1e9)),
    'MATERN': Matern(length_scale=0.8, length_scale_bounds=(1e-10, 100000.0), nu=0.5),
    'CONSTANT': ConstantKernel(0.01)
}
kernel_names = list(Kernels_dict.keys()) # [0'RBF', 1'PERIODIC', 2'RQ', 3'LINEAR', 4'MATERN', 5'CONSTANT']
kernel_idx  = [2, 3]
operation   = 'OR' # AND (x) or OR (+)
if operation == 'AND':
    filename    = os.path.join(save_path, '{}_{}-{}-{}-{}X{}_gp.pickle'.format(map_name, line_type, data_size, state_names[VARIDX], kernel_names[kernel_idx[0]], kernel_names[kernel_idx[1]]))
elif operation == 'OR':
    filename    = os.path.join(save_path, '{}_{}-{}-{}-{}+{}_gp.pickle'.format(map_name, line_type, data_size, state_names[VARIDX], kernel_names[kernel_idx[0]], kernel_names[kernel_idx[1]]))
GPy         = False
skl         = True

def load_data(CTYPE, map_name, VARIDX, data_path, data_name, line_type, xscaler=None, yscaler=None, test_mode=None):
    start_idx  = 0
    new_tracks = False
    test_path  = data_path
    test_name  = data_name
    if test_mode:
        if new_tracks:
            data_path   = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/')
            map_index   = 2
            map_name    = map_list[map_index]
            line_type   = 'centerline'
            ORIGINAL    = False
            if ORIGINAL:
                data_path += 'original/'
            elif line_type == 'centerline':
                data_path += 'centerline/'
            else:
                data_path += 'raceline/'
        test_path    += 'test/'
        test_name[0] += '_test'
        test_name[1] += '_test'
        data_dyn  = np.load(test_path + test_name[0].format(CTYPE, map_name, line_type) + '.npz')
        data_kin  = np.load(test_path + test_name[1].format(CTYPE, map_name, line_type) + '.npz')
        y_all = data_dyn['states'][start_idx: ,:] - data_kin['states'][start_idx: ,:]
        x = np.concatenate([
            
            data_kin['states'][start_idx: ,2].T.reshape(-1,1), # 𝛿
            data_dyn['states'][start_idx: ,3].T.reshape(-1,1), # v 
            data_dyn['states'][start_idx:, -2:], # 'ω', 'β'
            data_kin['inputs'][start_idx: ,:], # acc, Δ𝛿
            ],
            axis=1)
    else: # training mode
        data_dyn  = np.load(data_path + data_name[0].format(CTYPE, map_name, line_type) + '.npz')
        data_kin  = np.load(data_path + data_name[1].format(CTYPE, map_name, line_type) + '.npz')
        y_all = data_dyn['states'][start_idx:, :] - data_kin['states'][start_idx:, :]
        x = np.concatenate([
            
            data_kin['states'][start_idx:, 2].T.reshape(-1,1), # 𝛿
            data_dyn['states'][start_idx:, 3].T.reshape(-1,1), # v 
            data_dyn['states'][start_idx:, -2:], # 'ω', 'β'
            data_kin['inputs'][start_idx:, :], # acc, Δ𝛿
            ],
            axis=1)
    y = y_all[:,VARIDX].reshape(-1,1)
    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(x)
        yscaler.fit(y)
        return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
    else:
        return xscaler.transform(x), yscaler.transform(y)
    
# Training data & testing data
x_train, y_train, xscaler, yscaler = load_data(CTYPE, map_name, VARIDX, data_path, data_name, line_type, test_mode=False)
x_test, y_test = load_data(CTYPE, map_name, VARIDX, data_path, data_name, line_type, xscaler=xscaler, yscaler=yscaler, test_mode=True)

# train GP model -- default method
if VARIDX == 5:
    if operation == 'OR':
        kernel = Kernels_dict[kernel_names[kernel_idx[0]]] + Kernels_dict[kernel_names[kernel_idx[1]]]
    elif operation == 'AND':
        kernel = Kernels_dict[kernel_names[kernel_idx[0]]] * Kernels_dict[kernel_names[kernel_idx[1]]]
    model = GaussianProcessRegressor(
        alpha=1e-9, 
        kernel=kernel, 
        normalize_y=True,
        n_restarts_optimizer=10,
        )
if VARIDX == 6:
    if operation == 'OR':
        kernel = Kernels_dict[kernel_names[kernel_idx[0]]] + Kernels_dict[kernel_names[kernel_idx[1]]]
    elif operation == 'AND':
        kernel = Kernels_dict[kernel_names[kernel_idx[0]]] * Kernels_dict[kernel_names[kernel_idx[1]]]
    model = GaussianProcessRegressor(
        alpha=1e-9, 
        kernel=kernel, 
        normalize_y=True,
        n_restarts_optimizer=10,
        )
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print('####################################')
print('training time: %ss' %(end - start))        
print('final kernel: %s' %(model.kernel_))
print('####################################')

if SAVE_MODELS:
	with open(filename, 'wb') as f:
		pickle.dump((model, xscaler, yscaler), f)


#####################################################################
# # test GP model on training data
y_train_mu, y_train_std = model.predict(x_train, return_std=True)
y_train = yscaler.inverse_transform(y_train)
y_train_mu = yscaler.inverse_transform(y_train_mu)
y_train_std *= yscaler.scale_

MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

print('root mean square error: %s' %(np.sqrt(MSE)))
# print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_train.mean()))))
print('R2 score: %s' %(R2Score))
print('explained variance: %s' %(EV))

#####################################################################
# test GP model on validation data
y_test_mu, y_test_std = model.predict(x_test, return_std=True)
y_test = yscaler.inverse_transform(y_test)
y_test_mu = yscaler.inverse_transform(y_test_mu)
y_test_std *= yscaler.scale_

MSE = mean_squared_error(y_test, y_test_mu, multioutput='raw_values')
R2Score = r2_score(y_test, y_test_mu, multioutput='raw_values')
EV = explained_variance_score(y_test, y_test_mu, multioutput='raw_values')
print('####################################')
print('root mean square error: %s' %(np.sqrt(MSE)))
# print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_test.mean()))))
print('R2 score: %s' %(R2Score))
print('explained variance: %s' %(EV))

np.savez(
        path + 'test_y.npz',
        testY = y_test, 
        )
# #####################################################################
# plot results
plot_true_predicted_variance(
	y_train, y_train_mu, y_train_std, id='Training',
	ylabel='{} '.format(state_names[VARIDX]), xlabel='Sample Index'
	)

plot_true_predicted_variance(
	y_test, y_test_mu, y_test_std, id='Validation',
	ylabel='{} '.format(state_names[VARIDX]), xlabel='Sample Index'
	)
plt.savefig('GPfile.jpg', dpi=700)
plt.show()