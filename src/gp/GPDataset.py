import numpy as np
import pickle
import os
import tqdm
import json
import matplotlib.pyplot as plt
from src.quad_opt.quad_optimizer import QuadOptimizer
from src.quad_opt.quad import custom_quad_param_loader
from src.utils.utils import separate_variables, v_dot_q, quaternion_inverse, vw_to_vb

class GPDataset:
    def __init__(self, data_dir):
        """
            Load quad flight result and compile dataset to train GP
        """
        self.load_data(data_dir)

    def load_data(self, data_dir):
        """
            Load pickled Quadrotor Flight results
        """
        with open(os.path.join(data_dir, "results.pkl"), "rb") as fp:
            data = pickle.load(fp)
        with open(os.path.join(data_dir, "meta_data.json"), "rb") as fp:
            meta = json.load(fp)
        
        self.quad_name = meta["quad_name"]
        self.env = meta["env"]
        if "t_mpc" in meta.keys():
            self.mpc = True
            self.mhe = False
            # load MPC meta data
            # self.t_mpc = meta["t_mpc"]
            # self.n_mpc = meta["n_mpc"]
            # self.gt = meta["gt"]
            # self.with_gp = meta["with_gp"]

            # load flight data
            self.t = data["t"]
            self.dt = data["dt"]
            self.x_in = data["state_in"]
            self.error = data["error"]

            # Remove invalid entries (dt = 0)
            invalid = np.where(self.dt == 0)
            self.dt = np.delete(self.dt, invalid, axis=0)
            self.x_in = np.delete(self.x_in, invalid, axis=0)
            self.error = np.delete(self.error, invalid, axis=0
                               )
            # GP input and output
            self.train_in = self.x_in
            self.train_out = self.error
        else:
            self.mpc = False
            self.mhe = True
            # load MHE meta data
            # self.t_mhe = meta["t_mhe"]
            # self.n_mhe = meta["n_mhe"]
            # self.mhe_type = meta["mhe_type"]
            # self.with_gp = meta["with_gp"]

            # load flight data
            self.sensor_meas = data["sensor_meas"]
            # self.motor_thrust = data["motor_thrusts"]
            self.error = data["error"]
            # GP input and output
            self.train_in = self.sensor_meas
            self.train_out = self.error

    def get_train_ds(self, x_idx=None, y_idx=None):
        """
        Returns a Dataset to train GP to compensate of MHE/MPC model error
        return: [x, y] 
        """
        if x_idx is not None and y_idx is not None:
            return self.train_in[:, x_idx], self.train_out[:, y_idx]
        else:
            return self.train_in, self.train_out
    
