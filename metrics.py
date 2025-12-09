# GlaS metrics, translated from the official Matlab code:
# https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/evaluation_v6.zip
#

import numpy as np
import torch

from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


def ObjectHausdorff(S=None, G=None):
    """
    Computes the Object-level Hausdorff distance between two labeled segmentation masks S and G.
    For each object in S, finds the best matching object in G (by overlap or minimum Hausdorff distance),
    and vice versa, then averages the weighted Hausdorff distances. Used for evaluating segmentation quality
    at the object level, especially in biomedical image analysis.

    Args:
        S (ndarray): Segmentation mask (ground truth or prediction), integer labels.
        G (ndarray): Segmentation mask (ground truth or prediction), integer labels.

    Returns:
        float: Object-level Hausdorff distance between S and G.
    """
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaS = (S > 0).sum()
    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))

    temp1 = 0
    for iLabelS in range(len(listLabelS)):
        Si = S == listLabelS[iLabelS]
        intersectlist = G[Si]
        if intersectlist.any():
            indexGi = stats.mode(intersectlist).mode
            Gi = G == indexGi
        else:
            tempDist = np.zeros((len(listLabelG), 1))
            for iLabelG in range(len(listLabelG)):
                Gi = G == listLabelG[iLabelG]
                tempDist[iLabelG] = Hausdorff(Gi, Si)
            minIdx = np.argmin(tempDist)
            Gi = G == listLabelG[minIdx]
        omegai = Si.sum() / totalAreaS
        temp1 = temp1 + omegai * Hausdorff(Gi, Si)

    temp2 = 0
    for iLabelG in range(len(listLabelG)):
        tildeGi = G == listLabelG[iLabelG]
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = S == indextildeSi
        else:
            tempDist = np.zeros((len(listLabelS), 1))
            for iLabelS in range(len(listLabelS)):
                tildeSi = S == listLabelS[iLabelS]
                tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
            minIdx = np.argmin(tempDist)
            tildeSi = S == listLabelS[minIdx]
        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 = temp2 + tildeOmegai * Hausdorff(tildeGi, tildeSi)

    objHausdorff = (temp1 + temp2) / 2
    return objHausdorff


def Hausdorff(S=None, G=None, *args, **kwargs):
    """
    Computes the Hausdorff distance between two binary segmentation masks S and G.

    Args:
        S (ndarray): Binary segmentation mask (ground truth or prediction).
        G (ndarray): Binary segmentation mask (ground truth or prediction).

    Returns:
        float: Hausdorff distance between S and G. Returns 0 if both are empty, Inf if only one is empty.
    """
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))

    numS = len(listS)
    numG = len(listG)
    if numS == 0 and numG == 0:
        hausdorffDistance = 0
        return hausdorffDistance
    else:
        if numS == 0 or numG == 0:
            hausdorffDistance = np.Inf
            return hausdorffDistance

    y = np.where(S > 0)
    x = np.where(G > 0)

    x = np.vstack((x[0], x[1])).transpose()
    y = np.vstack((y[0], y[1])).transpose()

    nbrs = NearestNeighbors(n_neighbors=1).fit(x)
    distances, indices = nbrs.kneighbors(y)
    dist1 = np.max(distances)

    nbrs = NearestNeighbors(n_neighbors=1).fit(y)
    distances, indices = nbrs.kneighbors(x)
    dist2 = np.max(distances)

    hausdorffDistance = np.max((dist1, dist2))
    return hausdorffDistance


def F1score(S=None, G=None):
    """
    Computes the object-level F1 score between two labeled segmentation masks S and G.

    Args:
        S (ndarray): Segmentation mask (prediction)
        G (ndarray): Segmentation mask (ground truth)

    Returns:
        float: Object-level F1 score between S and G.
    """
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    numS = len(listS)
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))
    numG = len(listG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    tempMat = np.zeros((numS, 3))
    tempMat[:, 0] = listS
    for iSegmentedObj in range(numS):
        intersectGTObjs = G[S == tempMat[iSegmentedObj, 0]]
        if intersectGTObjs.any():
            intersectGTObjs_flat = np.delete(
                intersectGTObjs.flatten(), np.where(intersectGTObjs.flatten() == 0)
            )

            if len(intersectGTObjs_flat) == 0:
                maxGTi = 0
            else:
                maxGTi = stats.mode(intersectGTObjs_flat).mode
            tempMat[iSegmentedObj, 1] = maxGTi

    for iSegmentedObj in range(numS):
        if tempMat[iSegmentedObj, 1] != 0:
            SegObj = S == tempMat[iSegmentedObj, 0]
            GTObj = G == tempMat[iSegmentedObj, 1]
            overlap = np.logical_and(SegObj, GTObj)
            areaOverlap = overlap.sum()
            areaGTObj = GTObj.sum()
            if areaOverlap / areaGTObj > 0.5:
                tempMat[iSegmentedObj, 2] = 1

    TP = (tempMat[:, 2] == 1).sum()
    FP = (tempMat[:, 2] == 0).sum()
    FN = numG - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0

    score = (2 * precision * recall) / (precision + recall)

    return score


def ObjectDice(S, G):
    """
    Computes the object-level Dice coefficient between two labeled segmentation masks S and G.

    Args:
        S (ndarray): Segmentation mask (prediction)
        G (ndarray): Segmentation mask (ground truth)

    Returns:
        float: Object-level Dice coefficient between S and G.
    """
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    numS = len(listLabelS)
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))
    numG = len(listLabelG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    temp1 = 0
    totalAreaS = (S > 0).sum()
    for iLabelS in range(len(listLabelS)):
        Si = S == listLabelS[iLabelS]
        intersectlist = G[Si]
        if intersectlist.any():
            indexG1 = stats.mode(intersectlist).mode
            Gi = G == indexG1
        else:
            Gi = np.zeros(G.shape)

        omegai = Si.sum() / totalAreaS
        temp1 += omegai * Dice(Gi, Si)

    temp2 = 0
    totalAreaG = (G > 0).sum()
    for iLabelG in range(len(listLabelG)):
        tildeGi = G == listLabelG[iLabelG]
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = S == indextildeSi  # np logical and?
        else:
            tildeSi = np.zeros(S.shape)

        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 += tildeOmegai * Dice(tildeGi, tildeSi)

    return (temp1 + temp2) / 2


def Dice(A, B):
    """
    Computes the Dice coefficient between two binary masks A and B.
    The Dice coefficient measures the overlap between two sets.

    Args:
        A (ndarray): Binary mask.
        B (ndarray): Binary mask.

    Returns:
        float: Dice coefficient between A and B.
    """
    intersection = np.logical_and(A, B)
    return 2.0 * intersection.sum() / (A.sum() + B.sum())


def pixelwise_acc(pred: torch.Tensor, lbl: torch.Tensor) -> float:
    """
    Computes pixelwise accuracy between prediction and label tensors.
    Measures the proportion of correctly classified pixels in 4D tensors.

    Args:
        pred (torch.Tensor): Predicted labels tensor.
        lbl (torch.Tensor): Ground truth labels tensor.

    Returns:
        float: Pixelwise accuracy (correct pixels / total pixels).
    """
    pixel_num = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3]
    correct = (pred == lbl).sum().item()
    return correct / pixel_num


def averaged_hausdorff_distance(points_a, points_b, max_distance=np.inf):
    """
    Compute the Averaged Hausdorff Distance (AHD) between two unordered sets of points.
    The function is symmetric and does not support batches (squeeze your inputs first).

    Args:
        points_a (array-like): Each row/element is an N-dimensional point.
        points_b (array-like): Each row/element is an N-dimensional point.
        max_distance (float): Maximum AHD to return if any set is empty. Default: np.inf.

    Returns:
        float: The Averaged Hausdorff Distance between points_a and points_b.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return max_distance

    arr_a = np.array(points_a)
    arr_b = np.array(points_b)

    assert arr_a.ndim == 2, f"points_a should be 2D, got {arr_a.ndim}D"
    assert arr_b.ndim == 2, f"points_b should be 2D, got {arr_b.ndim}D"
    assert arr_a.shape[1] == arr_b.shape[1], (
        f"Points in both sets must have the same number of dimensions, got {arr_a.shape[1]} and {arr_b.shape[1]}."
    )

    distance_matrix = pairwise_distances(arr_a, arr_b, metric="euclidean")
    avg_min_a_to_b = np.average(np.min(distance_matrix, axis=1))
    avg_min_b_to_a = np.average(np.min(distance_matrix, axis=0))

    return avg_min_a_to_b + avg_min_b_to_a
