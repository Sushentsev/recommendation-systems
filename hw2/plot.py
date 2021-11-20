from typing import Optional, List, Any, Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def plot_similar(data: np.ndarray, labels: np.ndarray, title: Optional[str] = None):
    data = PCA(n_components=2).fit_transform(data)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_classes = len(le.classes_)

    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    colors = [colors[label] for label in labels]

    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], color=colors)
    if title is not None:
        plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


def plot_scores(scores: List[Dict[str, Any]], metric_type: str, metric_title: str):
    scores = np.vstack([it["validation"][metric_type] for it in scores])
    iterations = list(range(0, 100, 10))
    mean = scores.mean(axis=0)
    std = scores.std(axis=0)

    plt.figure(figsize=(10, 7))
    plt.plot(iterations, mean)
    plt.fill_between(x=iterations, y1=mean - std, y2=mean + std, alpha=.5)

    plt.title(metric_title)
    plt.xlabel("Iterations")
    plt.ylabel("Metric")
