"""
    Hyperspectral image segmentation

    The script generates a set of parameters to run the HSI segmentation with
"""

import os
from msiplib.segmentation.hsi.hsi_ms import ms_segmentation

def create_args_dict():
    # choose parameter setting to run algorithm with.
    args = {
        # path to configuration file (str). if provided, overrides all following choices
        "config_path": None,
        # path to image file (str)
        "filename": None,
        # path to ground truth file (str)
        "gt_file": None,
        # number of segments to be sought (int)
        "k": 4,
        # regularization parameter, balances data and regularization term (float)
        "reg_par": 0.01,
        # if true, add additional segment to collect irregular points
        "irreg": False,
        # radius of balls used to find irregular segments (int)
        "r": 1,
        # epsilon to compute indicator values for irregular region (float)
        "irreg_eps": 1e-06,
        # indicator function (str)
        # choices: ['2', 'anisotropic-eps-inverse', 'anisotropic-eps-normal',
        # 'anisotropic-eps-discard', 'epsAMS',
        # 'kernelpca-eps-inverse', 'kernelpca-eps-normal', 'AMS1']
        "ind_func": "epsAMS",
        # epsilon to regularize indicator function (float)
        "ind_eps": 1e-06,
        # kernel function (str)
        # choices: ['gaussian', 'polynomial', 'chi-squared', 'laplacian', 'cosine', 'sigmoid', 'direct-summation',
        # 'weighted-summation', 'cross-information']
        "kernel": "direct-summation",
        # list of parameters for indicator function
        # if indicator function has several parameters, give them with a list [value1, value2,...]
        # rbf kernel: [gamma]
        # polynomial kernel: [gamma, c_0, d]
        # direct summation kernel: [window size spat features, rbf params, polynomial params]
        # weighted summation kernel: [window size spat features, weight mu in [0, 1], rbf params, polynomial params]
        # cross-information kernel: [window size spat features, rbf params, polynomial params]
        "ind_params": [20],
        # convexification of Mumford-Shah functional, for multiphase segmentation only Zach implemented (str)
        # for binary segmentation: ['zach', 'binaryUC', 'strongly-conv', 'strongly-conv-uc']
        "convexification": "zach",
        # type of primal dual hybrid gradient (PDHG) algorithm used for optimization (int)
        "pdhgtype": 1,
        # set the precision of computations (str). choices: ['float', 'double']
        "precision": "float",
        # maximum number of PDHG iterations (int)
        "max_iter": 1000,
        # maximum number of outer iterations (int)
        "outer_iter": 20,
        # threshold of stopping criterion of PDHG method (float)
        "stop_eps": 1e-06,
        # threshold of outer stopping criterion (float)
        "outer_stop_eps": 1e-06,
        # band selection method (str). choices: ['full', 'issc']
        "band_selection": "full",
        # parameter of band selection method (float)
        "band_select_param": 0.0001,
        # reduce dimensionality of data before processing it (bool)
        "reduce_dimensionality": False,
        # dimensionality reduction method (str). choices: ['pca', 'tsne', 'umap', 'isomap', 'mnf', 'ica']
        "dim_red_method": "mnf",
        # number of features to keep (float)
        "n_features": 8,
        # initialization method. choices: ['kmeans', 'random', 'dbscan', 'gmm', 'hierarchical', 'optics', 'birch', 'gt']
        "init": "kmeans",
        # should the dimensionality be reduced before initialization is applied? (bool)
        # useful when initialization method should run on reduced data, but the segmentation method on the full data
        "reduce_dim_init": False,
        # computation of estimates of segment means (str). choices: ['arithmetic', 'trimmed']
        # does not apply to epsAMS
        "means": "arithmetic",
        # computation of estimates of segment variances (str). choices: ['sdm', 'trimmed']
        # sdm: squared differences from mean
        # does not apply to epsAMS
        "variances": "sdm",
        # computation of estimates of principal components of the segments (str). choices: ['pca', 'mnf']
        # does not apply to epsAMS
        "components": "pca",
        # proportion of points trimmed at both sides when computing trimmed means or variances (float)
        "trim_proportion": 0.5,
        # seed for randomness to make results reproducible (int)
        "seed": 42,
        # ignore pixels that carry this label in the ground truth (int)
        "ignore_label": 0,
        # if true, ignore pixels with ignore label in computations (bool)
        "ignore_pixels": False,
        # if true, ignore pixels with ignore label in data term (bool)
        "ignore_pixels_data_term": False,
        # if true, try to solve empty segments when they occur (bool)
        "solve_empty_segment": False,
        # if true, save found indermediate segmentations (bool)
        "save_intermediate_segs": False,
        # if true, use GPU to run computations when available
        "use_gpu": True,
        # number of device to use if several GPUs are available
        "device": 0
    }
    return args

def main():
    """main function"""

    ############# Preparation ##############
    # set environment variables for the segmentation output directory and the directory containing the msiplib
    os.environ['OUTPUT_DIR'] = './output'
    os.environ['REPOS_DIR'] = '.'

    # get arguments dictionary with default settings
    args = create_args_dict()


    ############# Set your parameters ##############
    # set input file
    args["filename"] = "<path to input file>"

    # set number of segments
    args["k"] = 3

    # set regularization parameter lambda
    args["reg_par"] = 0.01

    ms_segmentation(args)

if __name__ == "__main__":
    main()
