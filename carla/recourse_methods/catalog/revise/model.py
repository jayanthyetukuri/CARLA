from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from carla.data.api import Data
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import (
    Dataloader,
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.recourse_methods.processing.counterfactuals import (
    check_counterfactuals,
    reconstruct_encoding_constraints,
)


class Revise(RecourseMethod):
    def __init__(self, mlmodel: MLModel, data: Data, hyperparams: Dict) -> None:
        """
        Initialisation of the REVISE recourse method.

        Restrictions
        ------------
        - Works currently only on Pytorch models
        - Only binary categorical features with and without one-hot-encoding

        Parameters
        ----------
        mlmodel: Black-box-model we want to explore
        data: Dataset to perform on
        hyperparams: Parameter for Revise method, with following possibilites
            {
                "data_name": str  name of the dataset,
                "lambda": float default: 0.5    Decides how similar the counterfactual is to the factual,
                "optimizer": str defaul: "adam" Optimizer for generation of counterfactuals,
                            only adam and rmsprop possible
                "lr": float default: 0.1    learning rate for Revise,
                "max_iter": int default 1000, number of iterations for Revise optimization,
                "target_class": List default: [0, 1]  List of one-hot-encoded target class,
                "binary_cat_features": bool default: True If true, the encoding of x is done by drop_if_binary
                "vae_params": Dict with parameter for VAE,
                    {
                        "d": 8,  # latent space
                        "D": test_factual.shape[1],  # input size
                        "H1": 512,
                        "H2": 256,
                        "train": False,
                        "lambda_reg": 1e-6,
                        "epochs": 5,
                        "lr": 1e-3,
                        "batch_size": 32,
                    }
            }
        """
        super().__init__(mlmodel)
        self.params = hyperparams

        self._target_column = data.target
        self._lambda = self.params["lambda"] if "lambda" in hyperparams.keys() else 0.5
        self._optimizer = (
            self.params["optimizer"] if "optimizer" in hyperparams.keys() else "adam"
        )
        self._lr = self.params["lr"] if "lr" in hyperparams.keys() else 0.1
        self._max_iter = (
            self.params["max_iter"] if "max_iter" in hyperparams.keys() else 1000
        )
        self._target_class = (
            hyperparams["target_class"]
            if "target_class" in hyperparams.keys()
            else [0, 1]
        )
        self._binary_cat_features = (
            hyperparams["binary_cat_features"]
            if "binary_cat_features" in hyperparams.keys()
            else True
        )

        df_enc_norm_data = self.encode_normalize_order_factuals(
            data.raw, with_target=True
        )

        vae_params = hyperparams["vae_params"]
        self.vae = VariationalAutoencoder(
            self.params["data_name"],
            vae_params["d"],
            df_enc_norm_data.shape[1] - 1,  # num features - target
            vae_params["H1"],
            vae_params["H2"],
        )

        if vae_params["train"]:
            self.vae = train_variational_autoencoder(
                self.vae,
                self._mlmodel.data,
                self._mlmodel.scaler,
                self._mlmodel.encoder,
                self._mlmodel.feature_input_order,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                self.vae.load(df_enc_norm_data.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        df_enc_norm_fact = self.encode_normalize_order_factuals(
            factuals, with_target=True
        )

        # prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            Dataloader(df_enc_norm_fact.values), batch_size=1, shuffle=False
        )

        # pay attention to categorical features
        encoded_feature_names = self._mlmodel.encoder.get_feature_names(
            self._mlmodel.data.categoricals
        )
        cat_features_indices = [
            df_enc_norm_fact.columns.get_loc(feature)
            for feature in encoded_feature_names
        ]

        list_cfs = []
        for query_instance, _ in test_loader:

            target = torch.FloatTensor(self._target_class).to(device)
            target_prediction = np.argmax(np.array(self._target_class))

            z = self.vae.encode(query_instance)[0].clone().detach().requires_grad_(True)

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
                # z.requires_grad = True
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            candidate_counterfactuals = []  # all possible counterfactuals
            # distance of the possible counterfactuals from the intial value -
            # considering distance as the loss function (can even change it just the distance)
            candidate_distances = []
            all_loss = []

            for idx in range(self._max_iter):
                cf = self.vae.decode(z)[0]
                cf = reconstruct_encoding_constraints(
                    cf, cat_features_indices, self.params["binary_cat_features"]
                )
                output = self._mlmodel.predict_proba(cf)[0]
                _, predicted = torch.max(output, 0)

                z.requires_grad = True
                loss = self.compute_loss(cf, query_instance, target)
                all_loss.append(loss)

                if predicted == target_prediction:
                    candidate_counterfactuals.append(
                        cf.cpu().detach().numpy().squeeze(axis=0)
                    )
                    candidate_distances.append(loss.cpu().detach().numpy())

                loss.backward()
                optim.step()
                optim.zero_grad()
                cf.detach_()

            # Choose the nearest counterfactual
            if len(candidate_counterfactuals):
                print("Counterfactual found!")
                array_counterfactuals = np.array(candidate_counterfactuals)
                array_distances = np.array(candidate_distances)

                index = np.argmin(array_distances)
                list_cfs.append(array_counterfactuals[index])
            else:
                print("No counterfactual found")
                list_cfs.append(query_instance.cpu().detach().numpy().squeeze(axis=0))

        cf_df = check_counterfactuals(self._mlmodel, list_cfs)

        return cf_df

    def compute_loss(self, cf_initialize, query_instance, target):

        loss_function = nn.BCELoss()
        output = self._mlmodel.predict_proba(cf_initialize)[0]

        # classification loss
        loss1 = loss_function(output, target)
        # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), 1)

        return loss1 + self._lambda * loss2
