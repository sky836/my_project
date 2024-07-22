import os

import numpy as np
from matplotlib import pyplot as plt

data_name = 'PEMS08'
data = np.load(os.path.join(fr'{data_name}/', "data.npz"))["data"].astype(np.float32)
x = data[..., 0]



