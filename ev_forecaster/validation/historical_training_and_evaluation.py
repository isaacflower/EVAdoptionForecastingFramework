from ev_forecaster.models.model import GPForecastingModel, JointGPForecaster
from ev_forecaster.model_utils.transforms import invprobit

# Defining multiple variable types
from typing import Union
Numeric = Union[float, int, np.ndarray, pd.Series]

# Data
import pandas as pd
import numpy as np

# Gaussian Processes
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_summary_fmt("notebook")
f64 = gpflow.utilities.to_default_float # convert to float64 for tfp to play nicely with gpflow

# Progress Bars
from tqdm.notebook import tqdm

class GPForecastValidator():
    def __init__(
            self,
            GPForecastingModelClass,
            region_evms_df: pd.DataFrame,
            region_neighbourhood_dict: dict,
            region_list: np.ndarray,
            t_n_range: range,
            future_regional_data_dict: dict | None,
            t_0: int = 2011,
            t_f: int = 2023,
        ):
        self.GPForecastingModelClass = GPForecastingModelClass
        self.region_evms_df = region_evms_df
        self.region_neighbourhood_dict = region_neighbourhood_dict
        self.region_list = region_list
        self.t_n_range = t_n_range
        self.future_regional_data_dict = future_regional_data_dict
        self.t_0 = t_0
        self.t_f = t_f
        self.forecasters_dict = None
        self.models_dict = None
        self.mean_dict = None
        self.samples_dict = None
        self.em_dict = None
        self.mae_dict = None
        self.nmae_dict = None
        self.me_dict = None
        self.nme_dict = None
        pass

    # === Core Methods ===

    def train_joint_forecasters(self) -> dict[int, dict[str, GPForecastingModel]]:
        """
        Trains JointGPForecaster models for multiple regions and historical horizons.

        Returns:
            forecasters_dict: { forecast_start_year :  region_name : JointGPForecaster instance }}
            models_dict: { forecast_start_year : { region_name : list[GPForecastingModel] instance } }
        """

        forecasters_dict = {}
        models_dict = {}

        for t_n in tqdm(self.t_n_range, desc="Iterating over forecast starting points"):
            t_dict = {'t_0': self.t_0, 't_n': t_n, 't_f': self.t_f}
            horizon_forecasters = {}
            horizon_models = {}

            for region in self.region_list:

                if self.future_regional_data_dict is None:
                    future_regional_data = None
                else:
                    future_regional_data = self.future_regional_data_dict[region]

                # Define shared kernel & likelihood for this region and horizon
                kernel = gpflow.kernels.RBF(
                    lengthscales=gpflow.Parameter(f64(10.0), prior=tfp.distributions.Gamma(f64(10.0), f64(1.0)), transform=tfp.bijectors.Softplus()),
                    variance=gpflow.Parameter(f64(0.3), prior=tfp.distributions.Gamma(f64(3.0), f64(10.0)), transform=tfp.bijectors.Softplus())
                )
                likelihood = gpflow.likelihoods.Gaussian(
                    variance=gpflow.Parameter(f64(0.02), prior=tfp.distributions.Gamma(f64(2.0), f64(100.0)), transform=tfp.bijectors.Softplus())
                )

                # Build and train using JointGPForecaster
                forecaster = JointGPForecaster(
                    GPForecastingModelClass=self.GPForecastingModelClass,
                    region=region,
                    region_neighbourhood_dict=self.region_neighbourhood_dict,
                    region_evms_df=self.region_evms_df,
                    t_dict=t_dict,
                    future_regional_data=future_regional_data,
                    kernel_prior=kernel,
                    likelihood_prior=likelihood
                )

                forecaster.train()

                horizon_forecasters[region] = forecaster
                horizon_models[region] = forecaster.models
            forecasters_dict[t_n] = horizon_forecasters
            models_dict[t_n] = horizon_models
        self.forecasters_dict = forecasters_dict
        self.models_dict = models_dict
        return models_dict

    def extract_forecast_means_and_samples(self):
        """
        Generate forecast means and samples for multiple forecast start points (t_n values).

        """
        mean_dict = {}
        samples_dict = {}

        for t_n in tqdm(self.t_n_range, desc="Forecasting across t_n"):
            mean_dict_t_n = {}
            samples_dict_t_n = {}

            t_dict = {'t_0': 2011, 't_n': t_n, 't_f': 2023}

            for lad, forecaster in tqdm(self.forecasters_dict[t_n].items(), desc=f"Processing t_n={t_n}"):
                data = self.region_neighbourhood_dict[lad]['ev_ms']
                data = data.loc[:, data.notna().any()]

                # Compute forecasts
                mean_df, var_df = forecaster.run_forecasts(t_dict)
                mean_dict_t_n[lad] = mean_df

                # Generate samples
                samples_dict_t_n[lad] = forecaster.generate_forecast_samples(t_dict)

            # Store results for this t_n
            mean_dict[t_n] = mean_dict_t_n
            samples_dict[t_n] = samples_dict_t_n
        
        self.mean_dict = mean_dict
        self.samples_dict = samples_dict

        return mean_dict, samples_dict
    
    def calculate_error_metrics(self) -> dict:
        em_dict = {}
        for t_n in tqdm(self.t_n_range, desc="Iterating over forecast starting points"):
            t_dict = {'t_0': self.t_0, 't_n': t_n, 't_f': self.t_f}
            em_dict_t_f = {}
            
            for t_f in range(t_n+1, t_dict['t_f']+1):
                em_df = pd.DataFrame(index=self.region_list, columns=['MAE', 'nMAE', 'ME', 'nME'])

                for region, forecast_mean_df in tqdm(self.mean_dict[t_n].items(), desc=f'Evaluating Regions'):
                    data = self.region_neighbourhood_dict[region]['ev_ms']
                    actual = data.loc[t_f, data.notna().any()]
                    predicted = forecast_mean_df.map(invprobit).loc[t_f]
                    group_mean = self.region_evms_df.loc[t_f, region]

                    em_df.loc[region, 'MAE'] = self._calc_mae(predicted, actual)
                    em_df.loc[region, 'nMAE'] = self._calc_nmae(predicted, actual, group_mean)
                    em_df.loc[region, 'ME'] = self._calc_me(predicted, actual)
                    em_df.loc[region, 'nME'] = self._calc_nme(predicted, actual, group_mean)
                
                em_dict_t_f[t_f] = em_df
            em_dict[t_n] = em_dict_t_f
        
        self.em_dict = em_dict
        self._restructure_error_metrics(em_dict)
        return em_dict
    
    # === Internal Helper Methods === 

    # MAE
    def _calc_mae(self, predicted: Numeric, actual: Numeric, dp: int = 5) -> float:
        ae = np.abs(predicted - actual)
        mae = np.mean(ae)
        return round(mae, dp)

    # Normalised MAE (My own metric)
    def _calc_nmae(self, predicted: Numeric, actual: Numeric, group_mean: Numeric, dp: int = 3) -> float:
        ae = np.abs(predicted - actual)
        norm_mae = np.mean(ae) / group_mean
        return round(norm_mae, dp)

    # Mean Error (ME) [For evaluating model bias]
    def _calc_me(self, predicted: Numeric, actual: Numeric, dp: int = 5) -> float:
        e = predicted - actual
        me = np.mean(e)
        return round(me, dp)

    # Normalised Mean Error (ME) [For evaluating model bias]
    def _calc_nme(self, predicted: Numeric, actual: Numeric, group_mean: Numeric, dp: int = 5) -> float:
        e = predicted - actual
        norm_me = np.mean(e) / group_mean
        return round(norm_me, dp)
    
    def _restructure_error_metrics(self, em_dict: dict):
        mae_dict = {}
        nmae_dict = {}
        me_dict = {}
        nme_dict = {}

        for h_f in range(1, self.t_f - self.t_n_range[0] + 1):

            if self.t_f - h_f < list(self.em_dict.keys())[-1]:
                t_n_upper = self.t_f - h_f + 1
            else:
                t_n_upper = list(self.em_dict.keys())[-1] + 1

            t_n_range = range(self.t_n_range[0], t_n_upper)

            mae_df = pd.DataFrame(index=self.region_list, columns=t_n_range)
            nmae_df = pd.DataFrame(index=self.region_list, columns=t_n_range)
            me_df = pd.DataFrame(index=self.region_list, columns=t_n_range)
            nme_df = pd.DataFrame(index=self.region_list, columns=t_n_range)

            for t_n in t_n_range:
                mae_df[t_n] = em_dict[t_n][t_n+h_f]['MAE']
                nmae_df[t_n] = em_dict[t_n][t_n+h_f]['nMAE']
                me_df[t_n] = em_dict[t_n][t_n+h_f]['ME']
                nme_df[t_n] = em_dict[t_n][t_n+h_f]['nME']
            
            mae_dict[h_f] = mae_df
            nmae_dict[h_f] = nmae_df
            me_dict[h_f] = me_df
            nme_dict[h_f] = nme_df
        
        self.mae_dict = mae_dict
        self.nmae_dict = nmae_dict 
        self.me_dict = me_dict
        self.nme_dict = nme_dict
        print("Error metrics stored as: mae_dict, nmae_dict, me_dict, nme_dict")