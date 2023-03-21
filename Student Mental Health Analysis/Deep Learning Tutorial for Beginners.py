# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np  # linear algebra

# filter warnings
warnings.filterwarnings('ignore')

# print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# load data set
x_l = np.load(r'C:\Users\Hassan\Desktop\ML I\dataset\X.npy')
Y_l = np.load(r'C:\Users\Hassan\Desktop\ML I\dataset\Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[255].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[888].reshape(img_size, img_size))
plt.axis('off')
plt.show()
