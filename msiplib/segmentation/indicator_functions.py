#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=not-an-iterable

""" Collection of indicator functions used for image segmentation """

import logging
import numpy as np
from numba import jit, prange
from scipy.ndimage import uniform_filter
from scipy.stats import trimboth, trim_mean
from spectral import calc_stats, noise_from_diffs, mnf
from msiplib.decomposition import pca

from msiplib.metrics import anisotropic_2norm as m_anisotropic_2norm


def compute_segment_mean(image, seg_mask, label):
    """
    computes mean feature vector of a segment in an image

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    """
    return np.mean(image[seg_mask == label], axis=0)


def epsAMS(image, seg_mask, label, eps, means, pcs, weights, tol=1e-05, max_iter=100, valid_mask=None):
    """
    computes for a given segment the indicator function based on the non-squared anisotropic 2-norm epsAMS that is
    regularized with 1 / epsilon to ensure invertibility of the covariance matrix

    Args:
        image: an image as a two- or three-dimensional tensor

        seg_mask: a matrix of the same size as the image with integer entries being the labels
                  of the corresponding pixels

        label: the label of the segment to be processed

        eps: regularization parameter used to take care of directions with very low standard deviation

        means: initial guess for mean feature vector

        pcs: initial guess for principal components

        weights: initial guess for weights (regularized standard deviations) of indicator function

        tol: tolerance for stopping criterion

        max_iter: maximum number of iterations to find mean and covariance

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

    Returns:
        a vector of the same dimension as the feature vectors of the pixels in the image being the mean feature of
        the segment
    """

    logger = logging.getLogger("indicator")

    # extract pixels belonging to segment
    if valid_mask is not None:
        # remove pixels that shall not contribute to computation of segment's mean, components and standard deviations
        valid_pixels = image[valid_mask]
        valid_segmentation_mask = seg_mask[valid_mask]
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]
    else:
        valid_pixels = image.reshape((image.shape[0] * image.shape[1], image.shape[-1]))
        valid_segmentation_mask = np.ravel(seg_mask)
        valid_segment_pixels = valid_pixels[valid_segmentation_mask == label]

    # initialize mean values for segment and allocate necessary memory
    tau = 1e-02
    seg_mean_old = np.full_like(means, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    weights_old = np.full_like(weights, np.finfo(np.float32).max, shape=image.shape[-1], dtype=image.dtype)
    seg_pcs_old = np.full_like(
        pcs, np.finfo(np.float32).max, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype
    )
    dists_inv = np.empty_like(image, shape=valid_segment_pixels.shape[0], dtype=image.dtype)
    pxs_centered = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    pxs_scaled = np.empty_like(image, shape=valid_segment_pixels.shape, dtype=image.dtype)
    cov = np.empty_like(image, shape=(image.shape[-1], image.shape[-1]), dtype=image.dtype)

    # TODO: Can we work with references here instead of copying the data? Would that make returning weights unnecessary?
    t_weights = weights[label].copy()
    seg_mean = means[label].copy()
    seg_pcs = pcs[label].copy()

    it = 0
    logger.info("Regularize 2-norm with tau: %s", tau)
    logger.info("Maximum number of fixed point iterations: %s", max_iter)
    logger.info("Stopping threshold: %s", tol)
    while (
        np.linalg.norm(seg_mean - seg_mean_old)
        + np.linalg.norm(t_weights - weights_old)
        + np.linalg.norm(seg_pcs - seg_pcs_old)
    ) > tol and it < max_iter:
        # store old values of mean, standard deviations and PCs
        np.copyto(seg_mean_old, seg_mean)
        np.copyto(weights_old, t_weights)
        np.copyto(seg_pcs_old, seg_pcs)

        # compute distances wrt to current mean, standard deviations and PCs and store the reciprocals
        dists_inv[...] = m_anisotropic_2norm(
            (valid_segment_pixels - seg_mean[np.newaxis]).T, seg_pcs, t_weights, squared=False, tau=tau
        )
        np.reciprocal(dists_inv, out=dists_inv)

        # compute estimate of mean
        np.multiply(valid_segment_pixels, dists_inv[np.newaxis].T / np.sum(dists_inv), out=pxs_scaled)
        np.sum(pxs_scaled, axis=0, out=seg_mean)

        # compute estimate of covariance
        np.subtract(valid_segment_pixels, seg_mean[np.newaxis], out=pxs_centered)
        np.divide(dists_inv, 2.0, out=dists_inv)
        np.multiply(pxs_centered, dists_inv[np.newaxis].T, out=pxs_scaled)
        np.matmul(pxs_scaled.T, pxs_centered, out=cov)
        np.divide(cov, valid_segment_pixels.shape[0], out=cov)

        # eigenvalue decomposition of cov
        seg_std, seg_pcs = np.linalg.eigh(cov)
        np.maximum(seg_std, 0.0, out=seg_std)
        np.sqrt(seg_std, out=seg_std)

        # compute the weights for anisotropic 2-norm with current iterates
        np.maximum(seg_std, eps, out=t_weights)
        np.reciprocal(t_weights, out=t_weights)

        it += 1

    logger.info("Number of iterations needed to find mean and covariance: %s", it)
    logger.info("Components with standard deviation smaller than epsilon: %s", np.sum(t_weights == 1 / eps))

    # compute logarithm of determinant of covariance matrix
    log_det_cov = -2 * np.sum(np.log(t_weights))

    return (
        (
            m_anisotropic_2norm(
                (image.reshape((image.shape[0] * image.shape[1], image.shape[2])) - seg_mean[np.newaxis]).transpose(),
                seg_pcs,
                t_weights,
                squared=False,
                tau=tau,
            ).reshape((image.shape[0], image.shape[1]))
            + log_det_cov
        ),
        seg_mean,
        seg_pcs,
        t_weights,
    )




def euclidean_norm(image, segmentation_mask, label, valid_mask=None):
    """
    computes for a given segment the indicator function based on the euclidean norm

    Args:
        image: an image as a two- or three-dimensional tensor

        segmentation_mask: a matrix of the same size as the image with integer entries being the labels of
                           the corresponding pixels

        label: the label of the segment of which the mean feature vector shall be computed

        valid_mask: a matrix of the same size as the image with an entry being true if the corresponding pixel
                    shall contribute to the computation of the segment mean

    Returns:
        a matrix of the size as the image where at entry the indicator value of the pixel with respect to the
        considered segment is stored
    """

    if valid_mask is not None:
        # remove pixels that shall not contribute to the computation of the segment's mean
        valid_pixels = image[valid_mask]
        valid_segmentation = segmentation_mask[valid_mask]
    else:
        valid_pixels = image
        valid_segmentation = segmentation_mask

    segment_mean = compute_segment_mean(valid_pixels, valid_segmentation, label)

    return np.sum(np.square(image - segment_mean), axis=2)

