import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch

from src.gp.GPyModelWrapper import GPyModelWrapper
from src.gp.gp_utils import *
from src.utils.DirectoryConfig import DirectoryConfig as DirConf
from src.gp.GPDataset import GPDataset
def train_MPC_gp(quad_name, model_type, trajectory_name, environment, gt, epoch, n_dense_points=None, n_sparse_points=None, n_induce=None):
    """
    Train GP models for MPC model compensation. The trained GP model provides acceleration corrections to the dynamic model for
    improved accuracy of model prediction. 
    The "Exact" model will be trained by training a "dense" model using higher 
    number of n_dense_points, and a "sparse" model with smaller n_sparse_points generated by this dense model. 
    The "Approx" model will be trained with n_dense_points 
    :param quad_name: Name of the quadrotor
    :type quad_name: string
    :param model_type: String value indicating the GP model type ["Exact" or "Approx"]
    :type model_type: string
    :param trajectory_name: The name of the trajectory that was executed to collect flight data
    :type trajectory_name: string
    :param environment: String value indicating the environment the quadrotor flight was executed for data collection
    :type environment: string
    :param gt: Boolean value to indicate whether groundtruth state measurements were used for flight execution.
    :type gt: Bool
    :param epoch: Number of training epochfor GP training
    :type epoch: Int
    :param n_dense_points: Integer value indicating number of points utilized for Exact Dense GP training
    :type n_dense_points: Int
    :param n_sparse_points: Integer value indicating number of points utilized for Exact Sparse GP training
    :type n_sparse_points: Int
    :param n_induce: Number of inducing points for Approx GP model types
    :type n_induce: Int
    """
    assert model_type in ["Exact", "Approx"]
    load_model = False

    # Load Dataset
    flight_name = "%s_mpc%s_%s"%(env, "_gt" if gt else "", quad_name)
    results_dir = os.path.join(DirConf.FLIGHT_DATA_DIR, flight_name, trajectory_name)
    x_features_idx = [7, 8, 9]
    y_features_idx = [7, 8, 9]
    gp_ds = GPDataset(results_dir)

    # Select data points to be used
    x_train = np.zeros((len(x_features_idx), n_dense_points))
    y_train = np.zeros((len(x_features_idx), n_dense_points))
    for i, (xi, yi) in enumerate(zip(x_features_idx, y_features_idx)):
        train_in, train_out = gp_ds.get_train_ds(xi, yi)
        selected_idx =  distance_maximizing_points_1d(train_in, n_dense_points)
        x_train[i, :] = np.squeeze(train_in[selected_idx])
        y_train[i, :] = np.squeeze(train_out[selected_idx])
    x_train = torch.Tensor(x_train.T)
    y_train = torch.Tensor(y_train.T)

    # Create GP Model
    env = "gz" if environment == "gazebo" else "rt"
    model_name = "%s%s_%s_%s_mpc_%d_%d_%s"%("d" if model_type=="Exact" else "a", 
                                            str(n_induce) if model_type=="Exact" else "",
                                            env, 
                                            quad_name, 
                                            n_dense_points, 
                                            epoch, 
                                            model_type)
    gp_model = GPyModelWrapper(model_type, model_name, load=load_model, x_features=x_features_idx, y_features=y_features_idx)
    gp_model.train(x_train, y_train, epoch, induce_num=n_induce)

    if model_type == "Exact":
        # Select sparse data points to be used
        x_train = np.zeros((len(x_features_idx), n_sparse_points))
        y_train = np.zeros((len(x_features_idx), n_sparse_points))
        for i, (xi, yi) in enumerate(zip(x_features_idx, y_features_idx)):
            train_in, _ = gp_ds.get_train_ds(xi, yi)
            selected_idx =  distance_maximizing_points_1d(train_in, n_sparse_points, dense_gp=gp_model)
            x_train[i, :] = np.squeeze(train_in[selected_idx])
        x_train = x_train.T
        _, y_train = gp_model.predict(x_train, skip_variance=True)
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        # Create Dense GP Model
        env = "gz" if environment == "gazebo" else "rt"
        s_model_name = "s_%s_%s_mpc_%d_%d_%s"%(env, 
                                               quad_name, 
                                               n_sparse_points, 
                                               epoch, 
                                               model_type)
        gp_model = GPyModelWrapper(model_type, s_model_name, load=load_model, x_features=x_features_idx, y_features=y_features_idx)
        gp_model.train(x_train, y_train, epoch, induce_num=n_induce, dense_model_name=model_name)
        
    return gp_model

def train_MHE_gp(quad_name, model_type, trajectory_name, environment, epoch, n_dense_points=None, n_sparse_points=None, n_induce=None):
    """
    Train GP models for D-MHE model compensation. The trained GP model provides acceleration corrections to the dynamic model for
    improved accuracy of model prediction. 
    The "Exact" model will be trained by training a "dense" model using higher 
    number of n_dense_points, and a "sparse" model with smaller n_sparse_points generated by this dense model. 
    The "Approx" model will be trained with n_dense_points 
    :param quad_name: Name of the quadrotor
    :type quad_name: string
    :param model_type: String value indicating the GP model type ["Exact" or "Approx"]
    :type model_type: string
    :param trajectory_name: The name of the trajectory that was executed to collect flight data
    :type trajectory_name: string
    :param environment: String value indicating the environment the quadrotor flight was executed for data collection
    :type environment: string
    :param epoch: Number of training epochfor GP training
    :type epoch: Int
    :param n_dense_points: Integer value indicating number of points utilized for Exact Dense GP training
    :type n_dense_points: Int
    :param n_sparse_points: Integer value indicating number of points utilized for Exact Sparse GP training
    :type n_sparse_points: Int
    :param n_induce: Number of inducing points for Approx GP model types
    :type n_induce: Int
    """
    assert model_type in ["Exact", "Approx"]
    load_model = False

    # Load Dataset
    flight_name = "%s_dmhe_%s"%(env, quad_name)
    results_dir = os.path.join(DirConf.FLIGHT_DATA_DIR, flight_name, trajectory_name)
    x_features_idx = [6, 7, 8]
    y_features_idx = [7, 8, 9]
    gp_ds = GPDataset(results_dir)
    
    # Select data points to be used
    x_train = np.zeros((len(x_features_idx), n_dense_points))
    y_train = np.zeros((len(x_features_idx), n_dense_points))
    for i, (xi, yi) in enumerate(zip(x_features_idx, y_features_idx)):
        train_in, train_out = gp_ds.get_train_ds(xi, yi)
        selected_idx =  distance_maximizing_points_1d(train_in, n_dense_points)
        x_train[i, :] = np.squeeze(train_in[selected_idx])
        y_train[i, :] = np.squeeze(train_out[selected_idx])
    x_train = torch.Tensor(x_train.T)
    y_train = torch.Tensor(y_train.T)

    # Create GP Model
    env = "gz" if environment == "gazebo" else "rt"
    model_name = "%s%s_%s_%s_mhe_%d_%d_%s"%("d" if model_type=="Exact" else "a", 
                                            str(n_induce) if model_type=="Exact" else "",
                                            env, 
                                            quad_name, 
                                            n_dense_points, 
                                            epoch, 
                                            model_type)
    gp_model = GPyModelWrapper(model_type, model_name, load=load_model, x_features=x_features_idx, y_features=y_features_idx, mhe=True)
    gp_model.train(x_train, y_train, epoch, induce_num=n_induce)

    # Train Sparse Model for Exact GP
    if model_type == "Exact":
        # Select sparse data points to be used
        x_train = np.zeros((len(x_features_idx), n_sparse_points))
        y_train = np.zeros((len(x_features_idx), n_sparse_points))
        for i, (xi, yi) in enumerate(zip(x_features_idx, y_features_idx)):
            train_in, _ = gp_ds.get_train_ds(xi, yi)
            selected_idx =  distance_maximizing_points_1d(train_in, n_sparse_points, dense_gp=gp_model)
            x_train[i, :] = np.squeeze(train_in[selected_idx])
        x_train = x_train.T
        _, y_train = gp_model.predict(x_train, skip_variance=True)
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        # Create Dense GP Model
        env = "gz" if environment == "gazebo" else "rt"
        s_model_name = "s_%s_%s_mpc_%d_%d_%s"%(env, 
                                               quad_name, 
                                               n_sparse_points, 
                                               epoch, 
                                               model_type)
        gp_model = GPyModelWrapper(model_type, s_model_name, load=load_model, x_features=x_features_idx, y_features=y_features_idx, mhe=True)
        gp_model.train(x_train, y_train, epoch, induce_num=n_induce, dense_model_name=model_name)
        
    return gp_model

if __name__ == "__main__":
    print("x")