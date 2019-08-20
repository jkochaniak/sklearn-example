import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from termcolor import cprint


MARKER_SIZE = 50
MARKERS = ['o', 's', '^', 'P', '*', '1']


def show(data, clusters=None, title=None, show_plot=True):
    """
    Create a matplotlib for the given data. Handles 2d and 3d labeled data as well as 2d unlabeled data.
    :param data: Any of:
        * A pandas DataFrame with columns "x", "y", and "label"
        * A pandas DataFrame with columns "x", "y", "z", and "label"
        * An array-like object containing 2d data without labels, i.e. shape (n_samples, 2)
    :param clusters: Optional. An array-like object containing the cluster labels for the data. Only used if the data
        passed is unlabeled. Each distinct cluster label will be assigned a marker. Must be of length n_samples of
        the data.
    :param title: The optional title of the plot
    :param show_plot: If True, the plot window will be shown. If False, the new plot is created but not shown
        (call plot.show() manually to show the window).
    """
    if clusters is not None:
        fig = plot_clusters(data, clusters, title=title)
    elif type(data) == pd.core.frame.DataFrame and 'label' in data.columns:
        if data.shape[1] == 3:
            fig = plot_2d_labeled(data, title=title)
        else:
            fig = plot_3d_labeled(data, title=title)
    else:
        fig = plot_2d_unlabeled(data, title=title)
    if show_plot:
        plt.show()
    return fig


def plot_3d_labeled(data, title='3d Labeled Data'):
    """
    Plots a 3d dimensional labeled graph. The labels are used as the markers for the points.
    :param data: A pandas DataFrame with columns "x", "y", "z", and "label"
    :param title: A title for the graph
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    groups = data.groupby('label')
    for group in groups:
        label = group[0]
        group_data = group[1]
        ax.scatter3D(group_data.x, group_data.y, group_data.z, s=MARKER_SIZE, marker=label)
    plt.title(title)
    return fig


def plot_2d_unlabeled(data, title='2d Unlabeled Data'):
    """
    Plots 2d unlabeled data. All points will be assigned the same marker and color.
    :param data: An array-like object of shape (n_samples, 2) containing 2d data points
    :param title: A title for the graph
    """
    fig = plt.figure()
    ax = plt.axes()
    if type(data) == pd.core.frame.DataFrame:
        ax.scatter(data.x, data.y, marker='o', color='gray')
    else:
        ax.scatter(data[:, 0], data[:, 1], marker='o', color='gray')
    plt.title(title)
    return fig


def plot_2d_labeled(data, title='2d Labeled Data'):
    """
    Plots 2d labeled data. The labels are used as the markers for the points.
    :param data: A pandas DataFrame with columns "x", "y", and "label"
    :param title: A title for the graph
    """
    fig = plt.figure()
    ax = plt.axes()
    groups = data.groupby('label')
    for group in groups:
        label = group[0]
        group_data = group[1]
        ax.scatter(group_data.x, group_data.y, s=MARKER_SIZE, marker=label)
    plt.title(title)
    return fig


def plot_clusters(points, clusters, title='Clusters'):
    """
    Plots 2d unlabeled data, assigning markers based on the "clusters" argument.
    :param points: An array-like object of shape (n_samples, 2) containing 2d data points
    :param clusters: An array of length n_samples containing cluster identifiers for each point. Each distinct cluster
        identifier will be assigned a unique marker
    :param title: A title for the graph
    """
    labels = np.array([MARKERS[c] for c in clusters])
    data = pd.DataFrame(data=points, columns=['x', 'y'])
    data['label'] = labels
    return plot_2d_labeled(data, title=title)


def on_click_fig(fig, classifier, event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    point = [ix, iy]
    result = classifier.predict([point])[0]
    cprint(f'\nPredicted category "{categories[result]}" for point (x={point[0]:.2f}, y={point[1]:.2f})',
           attrs=['bold'])
    cprint('Click any point to predict its category, or close the plot window to continue.', attrs=['dark'])


def cluster_data():
    num_clusters_str = input("\nEnter number of clusters (up to 6), or type 'c' to continue: ")
    if num_clusters_str.lower() == 'c':
        plt.close('all')
        return
    try:
        num_clusters = int(num_clusters_str)
        if num_clusters > 6:
            print("I only have 6 markers, can't create more than 6 clusters.")
            cluster_data()
        if num_clusters < 1:
            print('Cluster number must be >= 1')
            cluster_data()
        print(f'\nClustered unlabeled data into {num_clusters} clusters')
        clusters = KMeans(n_clusters=num_clusters).fit_predict(unlabeled)
        show(unlabeled, clusters, f'{num_clusters} clusters', show_plot=False)
        plt.show(block=False)
        cluster_data()
    except ValueError:
        cluster_data()


if __name__ == '__main__':
    categories = {
        's': 'square',
        'o': 'circle',
        '^': 'triangle'
    }
    data3d = pd.read_csv('../sklearn-example/data/xyz-example.csv')
    labeled = pd.read_csv('../sklearn-example/data/xy-example.csv')
    classifier = MultinomialNB()

    print('\nTraining classifier on 2d labeled data')
    print('...')
    classifier.fit(labeled.values[:, :2], labeled.label)

    print('\nClick any point to predict its category, or close the plot window to continue.')
    fig = show(labeled, title='2d Labeled Data for Supervised Learning', show_plot=False)

    def on_click(event):
        on_click_fig(fig, classifier, event)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    unlabeled = labeled.values[:, :2]
    cprint('\nShowing 2d unlabeled data that will be clustered', attrs=['bold'])
    show(unlabeled, title='2d Unlabeled Data for Unsupervised Learning', show_plot=False)
    plt.show(block=False)

    cluster_data()

    print('\nShowing 3d labeled data')
    print('Close the plot window to exit')
    show(data3d, title='3d Labeled Data for Supervised Learning')

