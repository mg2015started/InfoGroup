import numpy as np
import pandas as pd

from algorithms.DecisionTree.plot_tree import createPlot


def load_dataset(file_name):
    """
    load csv dataset
    @param file_name: string
    @return: dataset: DataFrame
    """
    dataset = pd.read_csv(file_name, delimiter='\t')
    features = dataset.columns
    label_name = features[-1]
    feature_name = features[: -1]
    return dataset, feature_name, label_name


def cal_info_gain(dataset, feature_name, label_name):
    """
    cal the info gain g(dataset, feature)
    @param dataset: DataFrame
    @param feature: string
    @return: info gain: float
    """
    dataset_size = len(dataset)

    # cal H(D)
    groups = dataset.groupby(label_name)
    probs = groups.size() / dataset_size
    dataset_entropy = - np.sum(probs * np.log2(probs))

    # cal H(D|A)
    groups = dataset.groupby(feature_name)
    probs = groups.size() / dataset_size
    inner_entropy = groups[label_name].agg(
        lambda x: np.sum(- x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True))))
    dataset_condition_entropy = np.sum(probs * inner_entropy)

    # cal gain(D, A)
    info_gain = dataset_entropy - dataset_condition_entropy
    # print (feature_name, info_gain, dataset_entropy, dataset_condition_entropy)

    return info_gain


def choose_best_feature(dataset, label_name):
    """
    choose the max info gain feature when construct tree
    @param dataset: DataFrame
    @return: best_feature: string
    """
    features = dataset.columns[:-1]
    max_info_gain = -10
    best_feature = None
    for feat in features:
        info_gain = cal_info_gain(dataset, feat, label_name)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feat

    return best_feature


def create_tree(dataset, label_name):
    """
    create decision tree
    @param dataset: DataFrame
    @return: decision_tree: dict
    """
    # dataset has only one label, return that label
    if np.all(dataset[label_name].iloc[0] == dataset[label_name]):
        return dataset[label_name].iloc[0]

    # if only left one feature, return the major label
    if len(dataset.columns) == 2:
        return dataset.groupby(label_name).size().idxmax()

    # begin make tree
    best_feature = choose_best_feature(dataset, label_name)
    groups = dataset.groupby(best_feature)
    decision_tree = {best_feature: {}}
    for feature_value, dataset in groups:
        dataset = dataset.copy()
        # print(dataset)
        dataset = dataset.drop(columns=[best_feature])
        decision_tree[best_feature][feature_value] = create_tree(dataset, label_name)

    return decision_tree


if __name__ == '__main__':
    dataset, feature_name, label_name = load_dataset("buy_motorbike.csv")
    decision_tree = create_tree(dataset, label_name)
    createPlot(decision_tree)
