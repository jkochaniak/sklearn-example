from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets


MARKER_SIZE = 50
MARKERS = ['o', 's', '^']

# Modified for 3d from https://github.com/matplotlib/matplotlib/issues/11155#issuecomment-385939618
def mscatter(x, y, z, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    sc = ax.scatter(x, y, z, **kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


iris = datasets.load_iris()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

labels = np.array([MARKERS[t] for t in iris.target])
data = pd.DataFrame(data=iris.data, columns=['x', 'y', 'z', 'c'])
data['label'] = labels

# See https://stackoverflow.com/a/31575664
img = mscatter(data.x, data.y, data.z, c=data.c, s=MARKER_SIZE, m=data.label, cmap=plt.hot(), ax=ax)
cbar = fig.colorbar(img)
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('sepal width (cm)')
ax.set_zlabel('petal length (cm)')
cbar.set_label('petal width (cm)')
plt.title('Iris dataset')
plt.show()