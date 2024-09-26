import numpy as np
import pickle
import os
import tqdm

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
        with open(os.path.join(data_dir, "train_dataset.pkl"), "rb") as fp:
            data = pickle.load(fp)
        with open(os.path.join(data_dir, "meta_data.json"), "rb") as fp:
            meta = pickle.load(fp)
        
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
            # self.dt = data["dt"]
            self.x_in = data["state_in"]
            # self.x_out = data["state_out"]
            # self.x_act = data["x_act"]
            # self.x_pred = data["x_pred"]
            # self.motor_thrust = data["input_in"]
            self.error = data["error"]
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
            # self.t = data["t"]
            # self.x_est = data["x_est"]
            # self.x_act = data["x_act"]
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
            return self.train_in[x_idx], self.train_out[y_idx]
        else:
            return self.train_in, self.train_out
    
