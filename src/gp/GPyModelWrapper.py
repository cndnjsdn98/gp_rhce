import torch
import gpytorch
import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from src.utils.utils import safe_mkdir_recursive
from src.gp.gpy_model import *
from utils.DirectoryConfig import DirectoryConfig as DirConfig

class GPyModelWrapper:
    """
    Class for storing GPy Models, likelihood and its necessary parameters. 
    """
    def __init__(self, model_type, model_name=None, load=False,
                 keep_train_data=False, 
                 x_features=[7,8,9], u_features=[], y_features=[7,8,9], 
                 mhe=False,
                 model_dir=None):
        """
        :param model_type: String describing the Type of the GPy Model
        :type model_type: string
        :param model_name: String value of the model name
        :type model_name: string 
        :param load: Boolean value to determine whether to load the existing model if model_name exists
        :type load: Boolean
        :param keep_train_data: True if wish to keep train_x and train_y of the GPy model
        :type keep_train_data: bool
        :param x_features: List of n regression feature indices from the quadrotor state indexing.
        :type x_features: list
        :param u_features: List of n' regression feature indices from the input state.
        :type u_features: list
        :param y_features: Index of output dimension being regressed as the time-derivative.
        :type y_features: list  
        """
        self.machine = 0 # 0 indicttes cpu and 1 indicates gpu
        self.model_type = model_type
        self.model_name = model_name
        self.mhe = mhe
        #  Get Model directory
        if model_dir is not None:
            self.gp_model_dir = os.path.join(model_dir, model_name)
        else:
            self.gp_model_dir = os.path.join(DirConfig.GP_MODELS_DIR, self.model_type, model_name)
        
        # Check if a model exists in model_name
        if load and model_name is not None:
            if os.path.exists(os.path.join(self.gp_model_dir, "gpy_config.json")):
                # If the provided model name exists then load that model
                self.load(keep_train_data=keep_train_data)
        else:
            # Create the directory to save the gpy model and the meta data in
            try:
                safe_mkdir_recursive(self.gp_model_dir, overwrite=True)
            except:
                print("OVERWRITING EXISTING GP MODEL")
                
            # Else Set up Model paramters as provided
            self.x_features = x_features
            self.u_features = u_features
            self.y_features = y_features
            self.n_dim = len(x_features)
            self.model = None
            self.likelihood = None

    def train(self, train_x, train_y, train_iter, 
              sig_n=None, sig_f=None, l=None, 
              induce_num=None, induce_points=None,
              dense_model_name=None):
        """
        Trains the GPy Model with the given input training dataset.
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param train_iter: Integer value describing number of iterations
        for training the GPy model
        :type train_iter: integer
        :param sig_n: list of Prior noise variance
        :type sig_n: list
        :param sig_f: list of Data noise variance 
        :type sig_f: list
        :param l: list of length scale matrix
        :type l: list
        :param induce_num: Integer value for describing the number of inducing points 
        for variational GPy Models.
        :type induce_num: integer
        :param induce_points: Array of points describing the inducing locations
        for variational GPy Models.
        :type induce_points: list
        :param dense_model_name: Name of the dense model used to predict training data
        :type dense_model_name: string
        """
        if self.model_type == "Exact":
            self.train_and_save_Exact_model(train_x, train_y, train_iter, 
                                            sig_n=sig_n, sig_f=sig_f, l=l, 
                                            dense_model_name=dense_model_name)
        elif self.model_type == "Approx":
            self.train_and_save_Approx_model(train_x, train_y, train_iter,
                                             induce_num=induce_num, 
                                             induce_points=induce_points)
        self.machine = 1

    def predict(self, input, skip_variance=False):
        """
        Returns the predicted mean and variance value of the GPy model for 
        the given input value.
        :param input: A n_dim x N Array of inputs to be predicted
        :type input: torch.Tensor
        :return Array of predicted mean and variance value for the given input value. 
        """
        cov_pred = np.zeros(input.shape)
        mean_pred = np.zeros(input.shape)

        # If the input is an array with only single prediction points
        if input.ndim == 1:
            num_dim = 3
            test_x = {}
            for i in range(len(input)):
                test_x[i] = torch.Tensor([input[i]]).cuda()
        else:
            num_dim = input.shape[1]
            test_x = torch.Tensor(input).cuda()
        
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.memory_efficient(),  \
            gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition = False), \
            gpytorch.settings.debug(state=False), gpytorch.settings.max_cg_iterations(10),\
            gpytorch.settings.detach_test_caches(), \
            gpytorch.settings.eval_cg_tolerance(0.1), gpytorch.settings.max_root_decomposition_size(50),\
            gpytorch.settings.skip_posterior_variances(skip_variance):
            for i, model, likelihood in zip(range(num_dim), self.model.values(), self.likelihood.values()):
                if input.ndim > 1:
                    prediction = likelihood(model(test_x[:, i]))
                    cov_pred[:, i] = prediction.variance.cpu().detach().numpy() # Bring back on to CPU 
                    mean_pred[:, i] = prediction.mean.detach().cpu().numpy()
                else:
                    prediction = likelihood(model(test_x[i]))
                    cov_pred[i] = prediction.variance.cpu().detach().numpy()
                    mean_pred[i] = prediction.mean.detach().cpu().numpy()
                del prediction
                torch.cuda.empty_cache()
    
        del test_x
        if input.ndim == 1:
            return [cov_pred], [mean_pred]
        else:
            return cov_pred, mean_pred

    
    def load(self, keep_train_data = False):
        """
        Loads a pre-trained model from the specified directory, contained in a .pth file of GPy_Torch model 
        and json file of the configuration. 
        :param keep_train_data: True if wish to keep train_x and train_y of the GPy model
        :type keep_train_data: bool
        :return: a dictionary with the recovered model and the gp configuration
        """
        # Load Metadata
        f = open(os.path.join(self.gp_model_dir, "gpy_config.json"))
        gp_config = json.load(f)
        with open(os.path.join(self.gp_model_dir, "train_dataset.pkl"), "rb") as fp:
            train_dataset = pickle.load(fp)
        train_x = train_dataset["train_x"]
        train_y = train_dataset["train_y"]
        num_tasks = len(gp_config["x_features"])
        num_inputs = len(gp_config["y_features"])
        
        self.x_features = gp_config["x_features"]
        self.y_features = gp_config["y_features"]
        self.u_features = gp_config["u_features"]
        self.n_dim = len(self.x_features)
        self.mhe = json.loads(gp_config["mhe"].lower()) if "mhe" in gp_config else False

        #  There are models that haven't implemented prior parameters
        try:
            l = gp_config["l"]
            sig_n = gp_config["sig_n"]
            sig_f = gp_config["sig_f"]
        except:
            l = None
            sig_n = None
            sig_f = None

        # Load GPy model
        # If models are saved in a dictionary have to load each models separately and 
        # Add it to the dictionary of models and likelihoods
        model_dict = {}
        likelihood_dict = {}
        for i, idx in enumerate(gp_config["x_features"]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            if self.model_type == "Approx":
                model = ApproximateGPModel(torch.linspace(0, 1, gp_config["induce_num"]))
            elif self.model_type == "Exact":
                if sig_n is not None:
                    model = ExactGPModel(train_x[:, i], train_y[:, i], likelihood, 
                                                l=l[i], sig_n=sig_n[i], sig_f=sig_f[i])
                else:
                    model = ExactGPModel(train_x[:, i], train_y[:, i], likelihood)
            # load state_dict
            state_dict = torch.load(os.path.join(self.gp_model_dir, "gpy_model_" + str(idx) + ".pth"))
            model.load_state_dict(state_dict)
            model_dict[idx] = model.cuda().eval()
            likelihood_dict[idx] = likelihood.cuda().eval()
        model = model_dict
        likelihood = likelihood_dict    

        # Construct B_x(Output Selection matrix) and B_z(Input Selection matrix) matrix
        if self.mhe:
            B_z = np.zeros((10, num_tasks))
            B_x = np.zeros((13, num_inputs))
        else:
            B_z = np.zeros((13, num_tasks))
            B_x = np.zeros((13, num_inputs))

        for i, idx in enumerate(gp_config["x_features"]):
            B_z[idx, i] = 1
        for i, idx in enumerate(gp_config["y_features"]):
            B_x[idx, i] = 1
        
        # Save to Dictionary
        self.model = model
        self.likelihood = likelihood
        if keep_train_data:
            self.train_x = train_x
            self.train_y = train_y
        self.B_x = B_x
        self.B_z = B_z
        self.machine = 1

    def train_exact_model(self, train_x, train_y, train_iter, 
                          sig_f=None, sig_n=None, l=None):
        """
        Takes in training data and training parameters and trains an approximate GPy Model.
        Returns GPy model and its likelihood 
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :param train_y: torch.Tensor
        :param induce_num: Integer value of number of points to be induced for approximate GPy Model
        :param train_iter: Integer value of number of training iterations
        :return Trained GPy Model and Likelihood
        """
        # Set up GPy Model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        model = ExactGPModel(train_x, train_y, likelihood, 
                        sig_f=sig_f, sig_n=sig_n, l=l).cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        
        # Set up Optimizer and objective function
        objective_function = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.05)

        # Train
        model.train()
        likelihood.train()
        for i in range(train_iter):
            output = model(train_x)
            loss = -objective_function(output, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iter, loss.item()))

        model.eval()
        likelihood.eval()
        return model, likelihood

    def train_approximate_model(self, train_x, train_y, 
                                induce_num, train_iter, 
                                inducing_points=None):
        """
        Takes in training data and training parameters and trains an approximate GPy Model.
        Returns GPy model and its likelihood 
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param induce_num: Integer value of number of points to be induced for approximate GPy Model
        :param train_iter: Integer value of number of training iterations
        :return Trained GPy Model and Likelihood
        """
        # Set up GPy Model
        if inducing_points is None:
            inducing_points = torch.linspace(min(train_x), max(train_x), induce_num)

        learn_induce_points = inducing_points == None
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = ApproximateGPModel(inducing_points,learn_inducing_points=learn_induce_points).cuda()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        
        # Set up Optimizer and objective function
        objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data = train_y.numel())
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

        # Train
        model.train()
        likelihood.train()
        for i in range(train_iter):
            output = model(train_x)
            loss = -objective_function(output, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iter, loss.item()))
        model.eval()
        likelihood.eval()
        return model, likelihood

    def train_and_save_Exact_model(self, train_x, train_y, train_iter, 
                                   sig_n=None, sig_f=None, l=None, 
                                   dense_model_name=None):
        """
        Trains Exact GPy Model. 
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param train_iter: Number of iterations training the GPy Model.
        :type train_iter: integer
        :param sig_n: list of Prior noise variance
        :type sig_n: list
        :param sig_f: list of Data noise variance 
        :type sig_f: list
        :param l: list of length scale matrix
        :type l: list
        :param dense_model_name: Name of the dense model used to predict training data
        :type dense_model_name: string
        """
        model_dict = {}
        likelihood_dict = {}
        tic = time.time()
        # Train GPy Model
        for i, idx in enumerate(self.x_features):
            print("########## BEGIN TRAINING idx {} ##########".format(idx))
            if sig_n is not None:
                model, likelihood = self.train_exact_model(train_x[:, i], train_y[:, i], train_iter,
                    sig_f=sig_f[i], sig_n=sig_n[i], l=l[i])
            else: 
                model, likelihood = self.train_exact_model(train_x[:, i], train_y[:, i], train_iter)
            model_dict[idx] = model
            likelihood_dict[idx] = likelihood
        self.model = model_dict
        self.likelihood = likelihood_dict
        print("########## FINISHED TRAINING ##########")
        print("Elapsed time to train the GP: {:.2f}s".format(time.time() - tic))
        # Save GP Model
        train_length = train_x.shape[0]    
        # Save gpy model
        for i, idx in enumerate(self.x_features):
            model = model_dict[idx]
            print(f"############## idx {idx} params ##############")
            for param_name, param in model.named_parameters():
                print(f'Parameter name: {param_name:42} value = {param.item()}')
            torch.save(model.state_dict(), os.path.join(self.gp_model_dir, "gpy_model_" + str(idx) + ".pth"))  
            scripted_model = torch.jit.script(model)
            scripted_model.save(os.path.join(self.gp_model_dir, "scripted_gpy_model_" + str(idx) + ".pth"))
        print(self.model_name)
        # Save Meta data
        meta_data = {"x_features": self.x_features, 
                     "y_features": self.y_features, 
                     "u_features": self.u_features, 
                     "train_length": train_length, 
                     "train_iter": train_iter, 
                     "mhe": str(self.mhe),
                     "dense_model_name": dense_model_name if dense_model_name is not None else ""}
        if sig_n is not None:
            meta_data["sig_n"] = sig_n
            meta_data["sig_f"] = sig_f
            meta_data["l"] = l
        else:
            meta_data["sig_n"] = None
            meta_data["sig_f"] = None
            meta_data["l"] = None
        with open(os.path.join(self.gp_model_dir, "gpy_config.json"), 'w') as f:
            json.dump(meta_data, f)
        # Save Training Data
        train_dataset = {"train_x": np.array(train_x), "train_y": np.array(train_y)}
        with open(os.path.join(self.gp_model_dir, "train_dataset.pkl"), "wb") as fp:
            pickle.dump(train_dataset, fp)
        
    def train_and_save_Approx_model(self, train_x, train_y, train_iter,
                                    induce_num=20, induce_points=None):
        """
        Trains Approx GPy Model. If the induce_num is not given then it induces with
        20 points, and if the induce_num is given then it trains an
        approximate GPy Model with the given number of inducing points.
        If the induce_points are given then the approximate model will train for
        those inducing points, however if not provided it will learn inducing points.
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param train_iter: Number of iterations training the GPy Model.
        :type train_iter: integer
        :param induce_num: Number of inducing poitns for Approximate GPy Model.
        :type induce_num: integer
        :param induce_points: Array of inducing points for approximate GPy Model.
        :type induce_points: torch.Tensor
        """
        model_dict = {}
        likelihood_dict = {}
        tic = time.time()
        # Train GPy Model
        for i, idx in enumerate(self.x_features):
            print("########## BEGIN TRAINING idx {} ##########".format(idx))
            model, likelihood = self.train_approximate_model(train_x[:, i], train_y[:, i], induce_num, 
                                train_iter, inducing_points=induce_points[:, i] if induce_points is not None else None)
            model_dict[idx] = model
            likelihood_dict[idx] = likelihood
        self.model = model_dict
        self.likelihood = likelihood_dict
        print("########## FINISHED TRAINING ##########")
        print("Elapsed time to train the GP: {:.2f}s".format(time.time() - tic))
        # Save GPy Model
        train_length = train_x.shape[0]

        # Save GPy Models
        for i, idx in enumerate(self.x_features):
            model = model_dict[idx]
            torch.save(model.state_dict(), os.path.join(self.gp_model_dir, "gpy_model_" + str(idx) + ".pth"))
            scripted_model = torch.jit.script(model)
            scripted_model.save(os.path.join(self.gp_model_dir, "scripted_gpy_model_" + str(idx) + ".pth"))
        print(self.model_name)
        # Save meta data
        train_length = train_x.shape[1]
        u_features = []
        meta_data = {"x_features": self.x_features, "y_features": self.y_features, "u_features": self.u_features, \
                        "train_length": train_length, "train_iter": train_iter, "induce_num":induce_num, "mhe": str(self.mhe)}
        with open(os.path.join(self.gp_model_dir, "gpy_config.json"), 'w') as f:
            json.dump(meta_data, f)
        # Save the trianing data
        train_dataset = {"train_x": np.array(train_x), "train_y": np.array(train_y)}
        with open(os.path.join(self.gp_model_dir, "train_dataset.pkl"), "wb") as fp:
            pickle.dump(train_dataset, fp)
    
    def cpu(self):
        """
        Switch models to CPU if they are on GPU
        """
        try:
            self.model = self.model.cpu()
            self.likelihood = self.likelihood.cpu()
            self.machine = 0
        except:
            print("Model already on CPU")

    def gpu(self):
        """
        Switch models to GPU if they are on CPU
        """
        try:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        except:
            print("Model already on GPU")

    def get_Bx(self):
        return self.B_x
    
    def get_Bz(self):
        return self.B_z

    def get_x_features(self):
        return self.x_features
    
    def get_y_features(self):
        return self.y_features

    def get_model_type(self):
        return self.model_type

    def get_u_features(self):
        return self.u_features

    def get_model_name(self):
        return self.model_name
    
    def get_x_train(self):
        return self.train_x

    def get_y_train(self):
        return self.train_y
    
    def get_model_directory(self):
        return self.gp_model_dir