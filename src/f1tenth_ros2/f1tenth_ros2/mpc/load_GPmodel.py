import _pickle as pickle
import numpy as np
import time

data_dyn = np.load("/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/raceline/f1tenth-downsample-DYN-PP-Sepang_raceline_ED.npz")
x = np.concatenate([
            data_dyn['inputs'][0:, :], # acc, Δ𝛿
            data_dyn['states'][0:, 2].T.reshape(-1,1), # 𝛿
            data_dyn['states'][0:, 3].T.reshape(-1,1), # v 
            ],
            axis=1)
start = time.time()
with open('/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/gp_models/centerline/Shanghai_centerline-downsample-ω-RBF+LINEAR_gp.pickle', 'rb') as f:
	(ωmodel, ωxscaler, ωyscaler) = pickle.load(f)
x_test = x[-100].reshape(1,-1)
ωmodel.predict(x_test)
# print()
end = time.time()
print('time: ', end-start)