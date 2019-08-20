import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB



MARKER_SIZE = 50
MARKERS = ['o', 's', '^', 'P', '*', '1']


def show(data, clusters=None, title=None, show_plot=True):
    if clusters is not None:
        plot_clusters(data, clusters, title=title)
    elif type(data) == pd.core.frame.DataFrame and 'label' in data.columns:
        if data.shape[1] == 3:
            plot_2d_labeled(data, title=title)
        else:
            plot_3d_labeled(data, title=title)
    else:
        plot_2d_unlabeled(data, title=title)
    if show_plot:
        plt.show()


def plot_3d_labeled(data, title='3d Labeled Data'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    groups = data.groupby('label')
    for group in groups:
        label = group[0]
        group_data = group[1]
        ax.scatter3D(group_data.x, group_data.y, group_data.z, s=MARKER_SIZE, marker=label)
    plt.title(title)



def plot_2d_unlabeled(data, title='2d Unlabeled Data'):
    fig = plt.figure()
    ax = plt.axes()
    if type(data) == pd.core.frame.DataFrame:
        ax.scatter(data.x, data.y, marker='o', color='gray')
    else:
        ax.scatter(data[:, 0], data[:, 1], marker='o', color='gray')
    plt.title(title)


def plot_2d_labeled(data, title='2d Labeled Data'):
    fig = plt.figure()
    ax = plt.axes()
    groups = data.groupby('label')
    for group in groups:
        label = group[0]
        group_data = group[1]
        ax.scatter(group_data.x, group_data.y, s=MARKER_SIZE, marker=label)
    plt.title(title)


def plot_clusters(points, clusters, title='Clusters'):
    labels = np.array([MARKERS[c] for c in clusters])
    data = pd.DataFrame(data=points, columns=['x', 'y'])
    data['label'] = labels
    plot_2d_labeled(data, title=title)


if __name__ == '__main__':
    categories = {
        's': 'square',
        'o': 'circle',
        '^': 'triangle'
    }
    data3d = pd.read_csv('../sklearn-example/data/xyz-example.csv')
    labeled = pd.read_csv('../sklearn-example/data/xy-example.csv')
    classifier = MultinomialNB()
    classifier.fit(labeled.values[:, :2], labeled.label)
    point = [5.5, 8.0]
    result = classifier.predict([point])[0]
    print(f'Predicted category "{categories[result]}" for point (x={point[0]}, y={point[1]})')
    point = [9.8, 5.8]
    result = classifier.predict([point])[0]
    print(f'Predicted category "{categories[result]}" for point (x={point[0]}, y={point[1]})')
    point = [6, 6]
    result = classifier.predict([point])[0]
    print(f'Predicted category "{categories[result]}" for point (x={point[0]}, y={point[1]})')
    point = [8, 3.5]
    result = classifier.predict([point])[0]
    print(f'Predicted category "{categories[result]}" for point (x={point[0]}, y={point[1]})')
    print('\nClose the plot window to continue...')
    show(labeled, title='2d Labeled Data')

    print('Close the plot window to continue...')
    show(data3d, title='3d Labeled Data')

    unlabeled = labeled.values[:, :2]
    clusters_2 = KMeans(n_clusters=2).fit_predict(unlabeled)
    clusters_3 = KMeans(n_clusters=3).fit_predict(unlabeled)
    clusters_5 = KMeans(n_clusters=5).fit_predict(unlabeled)
    show(unlabeled, clusters_2, '2 clusters', show_plot=False)
    show(unlabeled, clusters_3, '3 clusters', show_plot=False)
    show(unlabeled, clusters_5, '5 clusters', show_plot=False)
    print('Close the plot window to continue...')
    plt.show()


