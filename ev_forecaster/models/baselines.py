# Import modules
from ev_forecaster.model_utils.transforms import probit, invprobit

# Data
import pandas as pd
import numpy as np
seed = 42
rng = np.random.default_rng(seed)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
sns.set_context("paper")

# For Logistic Growth Model
from scipy.optimize import curve_fit

# Progress Bars
from tqdm.notebook import tqdm

# Defining multiple variable types
from typing import Union
Numeric = Union[float, int, np.ndarray, pd.Series, pd.DataFrame]

# Bootstrapping Confidence Intervals
from sklearn.utils import resample

# === Baseline Helper Classes ===

class BaselineForecaster:
    def __init__(self, model_class, scenarios: pd.DataFrame = None):
        """
        Initialize LADForecaster.

        Args:
            model_class: The forecasting model class to use (must implement .fit() and .predict()).
            scenarios (pd.DataFrame, optional): Regional or LAD-level scenario data for models that require it.
        """
        self.model_class = model_class
        self.scenarios = scenarios
        self.forecasts_dict = None
        self.errors_dict = None

    # === Core Methods ===

    def generate_forecasts(self, lad_lsoa_dict: dict, lad_array: np.ndarray, t_s: int, t_n_first: int, t_f_last: int) -> dict:
        """Generate forecasts across multiple horizons (t_n values)."""
        forecasts_dict = {}
        for t_n in tqdm(range(t_n_first, t_f_last), desc="Evaluating t_n years"):
            forecasts_dict[t_n] = self._forecast_all_lads(
                lad_lsoa_dict=lad_lsoa_dict,
                lad_array=lad_array,
                t_s=t_s,
                t_n=t_n,
                t_f_last=t_f_last
            )
        
        self.forecasts_dict = forecasts_dict

        return forecasts_dict

    # === Internal Helper ===
        
    def _get_training_window(self, t_s: int, t_n: int) -> np.ndarray:
        """Return training window as numpy array."""
        return np.arange(t_s, t_n + 1).reshape(-1, 1)

    def _get_forecast_window(self, t_n: int, t_f_last: int) -> np.ndarray:
        """Return forecast window as numpy array."""
        return np.arange(t_n + 1, t_f_last + 1).reshape(-1, 1)
    
    def _fit_model(self, X_train, y, scenario):
        """Handles model-specific fitting logic."""
        model = self.model_class()
        model_name = type(model).__name__

        if model_name == 'ScaledScenario':
            return model.fit(X_train, y, scenario)
        elif model_name == 'LogisticGrowthModel':
            model.learn_bounds(scenario)
            return model.fit(X_train, y)
        elif model_name == 'LinearRegression':
            X_train = X_train[-3:]
            y = y[-3:]
            return model.fit(X_train, y)
        else:
            return model.fit(X_train, y)

    def _forecast_lsoa(self, data: pd.DataFrame, X_train: np.ndarray, X_forecast: np.ndarray, scenario: pd.Series = None) -> pd.DataFrame:
        """Forecast all LSOAs within a LAD."""
        forecasts = pd.DataFrame(index=X_forecast.flatten(), columns=data.columns)

        for col in data.columns:
            y = data[col].loc[X_train.flatten()].values
            model = self._fit_model(X_train, y, scenario)
            forecasts[col] = model.predict(X_forecast)

        return forecasts

    def _forecast_all_lads(self, lad_lsoa_dict: dict, lad_array: np.ndarray, t_s: int, t_n: int, t_f_last: int) -> dict:
        """Forecast all LADs."""
        X_train = self._get_training_window(t_s, t_n)
        X_forecast = self._get_forecast_window(t_n, t_f_last)
        lad_forecasts = {}

        for lad in lad_array:
            data = lad_lsoa_dict[lad]['ev_ms']
            data = data.loc[:, data.notna().any()]
            scenario = self.scenarios[lad] if self.scenarios is not None else None
            lad_forecasts[lad] = self._forecast_lsoa(data, X_train, X_forecast, scenario)

        return lad_forecasts

class BaselineEvaluator():
    def __init__(self, lad_lsoa_dict: dict, lad_evms_df: pd.DataFrame, lad_array: np.ndarray, t_n_first: int, t_f_last: int, h_f_array: np.ndarray):
        self.lad_lsoa_dict = lad_lsoa_dict
        self.lad_evms_df = lad_evms_df
        self.lad_array = lad_array
        self.t_n_first = t_n_first
        self.t_f_last = t_f_last
        self.h_f_array = h_f_array
        
    # === Core Methods ===
    
    def evaluate_forecasts_over_time(self, forecasts_dict: dict) -> dict:
        """Evaluate all forecasts across time and LADs."""
        errors_dict = {}
        
        for t_n in tqdm(forecasts_dict.keys(), desc="Evaluating t_n years"):
            error_metrics_dict = {}

            for t_f in range(t_n + 1, self.t_f_last + 1):
                error_metrics_df = self._evaluate_all_lads_for_year(
                    forecasts_dict_tn=forecasts_dict[t_n],
                    lad_lsoa_dict=self.lad_lsoa_dict,
                    lad_evms_df=self.lad_evms_df,
                    lad_array=self.lad_array,
                    t_f=t_f
                )
                error_metrics_dict[t_f] = error_metrics_df

            errors_dict[t_n] = error_metrics_dict

        return errors_dict
    
    def compute_nae_over_horizons(self, forecasts_dict: dict, data_transformation = None) -> tuple[dict, dict]:
        """Compute overall and per-LAD nMAE for each forecast horizon."""
        nae_dict = {}

        for h_f in self.h_f_array:
            t_n_array = self._get_t_n_array(self.t_n_first, self.t_f_last, h_f)
            nae_all = []

            for lad in self.lad_array:
                lad_nae_series = self._compute_lad_nae_for_horizon(
                    lad, h_f, t_n_array,
                    forecasts_dict=forecasts_dict,
                    lad_lsoa_dict=self.lad_lsoa_dict,
                    lad_evms_df=self.lad_evms_df,
                    data_transformation=data_transformation
                )
                nae_all.append(lad_nae_series)
            nae_dict[h_f] = pd.concat(nae_all)

        return nae_dict
        
    # === Internal Helper ===
    
    def _calc_mae(self, predicted: Numeric, actual: Numeric, dp: int = 5) -> Numeric:
        if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame | pd.Series):
            ae = np.abs(actual.sub(predicted, axis=0))
        else:
            ae = np.abs(actual - predicted)

        if isinstance(ae, pd.DataFrame):
            mae = ae.mean(axis=1)
        elif isinstance(ae, pd.Series):
            mae = ae.mean() 
        else:
            mae = np.mean(ae)

        return mae.round(dp) if isinstance(mae, (pd.Series, pd.DataFrame)) else round(mae, dp)

    def _calc_me(self, predicted: Numeric, actual: Numeric, dp: int = 5) -> Numeric:

        if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame | pd.Series):
            e = - actual.sub(predicted, axis=0)
        else:
            e = predicted - actual
        
        if isinstance(e, pd.DataFrame):
            me = e.mean(axis=1)
        elif isinstance(e, pd.Series):
            me = e.mean() 
        else:
            me = np.mean(e)

        return me.round(dp) if isinstance(me, (pd.Series, pd.DataFrame)) else round(me, dp)
    
    def _calc_nmae(self, predicted: Numeric, actual: Numeric, group_mean: Numeric, dp: int = 5) -> float:
        mae = self._calc_mae(predicted, actual, dp)
        nmae = mae / group_mean
        return round(nmae, dp)
    
    def _calc_nme(self, predicted: Numeric, actual: Numeric, group_mean: Numeric, dp: int = 5) -> float:
        me = self._calc_me(predicted, actual, dp)
        nme = me / group_mean
        return round(nme, dp)

    def _evaluate_forecast_for_lad(self, forecasts_df: pd.DataFrame, actual_df: pd.DataFrame, group_means: pd.Series, t_f: int) -> dict:
        """Evaluate forecast errors for a single LAD at time t_f."""
        actual = actual_df.loc[t_f]
        predicted = forecasts_df.loc[t_f]
        group_mean = group_means.loc[t_f]

        return {
            'MAE': self._calc_mae(predicted, actual),
            'nMAE': self._calc_nmae(predicted, actual, group_mean),
            'ME': self._calc_me(predicted, actual),
            'nME': self._calc_nme(predicted, actual, group_mean),
        }

    def _evaluate_all_lads_for_year(self, forecasts_dict_tn: dict, lad_lsoa_dict: dict, lad_evms_df: pd.DataFrame, lad_array: np.ndarray, t_f: int) -> pd.DataFrame:
        """Evaluate all LADs' forecasts for a single forecast year t_f."""
        error_metrics_df = pd.DataFrame(index=lad_array, columns=['MAE', 'nMAE', 'ME', 'nME'])

        for lad in lad_array:
            forecasts_df = forecasts_dict_tn[lad]
            data = lad_lsoa_dict[lad]['ev_ms']
            data = data.loc[:, data.notna().any()]

            scores = self._evaluate_forecast_for_lad(
                forecasts_df=forecasts_df,
                actual_df=data,
                group_means=lad_evms_df[lad],
                t_f=t_f
            )

            for metric, score in scores.items():
                error_metrics_df.loc[lad, metric] = score
        
        return error_metrics_df
    
    def _calc_nae(self, forecast: pd.Series, actual: pd.Series, group_mean: float) -> pd.Series:
        """Calculate normalised absolute error for each LSOA."""
        ae = (forecast - actual).abs()
        return ae / group_mean


    def _get_t_n_array(self, t_n_first: int, t_f_last: int, h_f: int) -> np.ndarray:
        """Return array of valid forecast origin years for a given horizon."""
        return np.arange(t_n_first, t_f_last + 1 - h_f)


    def _compute_lad_nae_for_horizon(self, lad: str, h_f: int, t_n_array: np.ndarray, forecasts_dict: dict, lad_lsoa_dict: dict, lad_evms_df: pd.DataFrame, data_transformation = None) -> pd.Series:
        """Compute concatenated nMAE for a LAD across all origin years t_n."""
        nae_list = []

        for t_n in t_n_array:
            t_f = t_n + h_f
            if data_transformation is None:
                forecast = forecasts_dict[t_n][lad].loc[t_f]
            else:
                forecast = forecasts_dict[t_n][lad].map(data_transformation).loc[t_f]
            actual = lad_lsoa_dict[lad]['ev_ms'].loc[t_f]
            group_mean = lad_evms_df.loc[t_f, lad]

            nae_list.append(self._calc_nae(forecast, actual, group_mean))

        return pd.concat(nae_list)

class BaselineNMAEComparator():
    def __init__(self, iter_array: np.ndarray):
        self.iter_array = iter_array
        pass

    # === Core Methods ===

    def compute_bootstrap_results(self, nae_dict: dict, n_bootstraps: int = 1000) -> pd.DataFrame:
        """Compute bootstrap mean and confidence intervals for each LAD at fixed horizon h_f."""
        results = {'Metric': [f'nMAE', 'Lower CI', 'Upper CI']}

        for i in self.iter_array:
            nae_values = nae_dict[i]
            mean_nmae, lower, upper = self._bootstrap_ci(nae_values, n_bootstraps=n_bootstraps)
            results[i] = [mean_nmae, lower, upper]

        return pd.DataFrame(results).set_index('Metric')

    def gather_model_results(self, model_results: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Convert results into long-form DataFrame for combined plotting."""
        means = model_results.loc['nMAE']
        lower = model_results.loc['Lower CI']
        upper = model_results.loc['Upper CI']
        
        df = pd.DataFrame({
            'Horizon': means.index,
            'Model': model_name,
            'nMAE': means.values,
            'Lower CI': lower.values,
            'Upper CI': upper.values
        })

        df['yerr_lower'] = df['nMAE'] - df['Lower CI']
        df['yerr_upper'] = df['Upper CI'] - df['nMAE']
        
        return df

    def plot_combined_nmae(self, df: pd.DataFrame, save_file_path:str = None):
        """Plot grouped bar chart with CIs from combined DataFrame."""
        bar_width = 0.15
        h_f_array = np.arange(1, 6)
        models = df['Model'].unique()
        x = np.arange(len(h_f_array))

        cmap = plt.colormaps.get_cmap('Blues')

        fig, ax = plt.subplots(figsize=(len(models)*2.5, 5))

        # Plot each model as a separate bar group
        for j, model in enumerate(models):
            df_model = df[df['Model'] == model].sort_values('Horizon')
            means = df_model['nMAE'].values
            yerr_lower = df_model['yerr_lower'].values
            yerr_upper = df_model['yerr_upper'].values
            yerr = np.vstack([yerr_lower, yerr_upper])

            if model == 'GP':
                color = 'deeppink'
            else:
                color = cmap((j+0.5) / (len(models)))

            ax.bar(
                x + j * bar_width,
                means,
                width=bar_width,
                label=model,
                yerr=yerr,
                capsize=3,
                color=color,
                alpha=0.8
            )

        # Axis and formatting
        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(h_f_array)
        ax.set_xlabel('$h_f$', fontsize=16, labelpad=5)
        ax.set_ylabel('nMAE', fontsize=16, labelpad=5)
        ax.tick_params(labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)

        # Legend and layout
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(models), fontsize=14)
        plt.tight_layout()
        if save_file_path is not None:
            plt.savefig(save_file_path, bbox_inches="tight", pad_inches=0.2)
        plt.show()

    # === Internal Helper ===

    def _bootstrap_ci(self, data: pd.Series, n_bootstraps: int = 1000, ci: float = 95) -> tuple[float, float, float]:
        """Return mean and CI bounds for bootstrap resampled data."""
        boot_means = [np.mean(resample(data)) for _ in range(n_bootstraps)]
        lower, upper = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        return np.mean(data), lower, upper

# === Baseline Models ===

class ScaledScenario:
    def __init__(self):
        self.scenario = None
        self.weight = None
        self.fitted_bool = False
        self.forecast = None
        self.scenario_forecast = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, scenario: pd.Series):
        """
        Fit logistic growth model to training data.
        X_train and Y_train should be 1D arrays.
        """
        X_train = X_train.flatten()
        Y_train = Y_train.flatten()
        self.scenario = scenario

        scenario_start = self.scenario.loc[X_train[-1]]
        self.weight = Y_train[-1] / scenario_start 

        self.fitted_bool = True

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted_bool:
            raise RuntimeError("Model must be fitted before calling predict.")
        
        scenario_forecast = self.scenario.loc[X_test.flatten()[0]-1::]
        self.scenario_forecast = scenario_forecast
        forecast = self.weight * scenario_forecast
        self.forecast = forecast.loc[X_test.flatten()].values

        return self.forecast

class LogisticGrowthModel:
    def __init__(self):
        self.k_init = None  # Growth rate
        self.t0_init = None  # Midpoint
        self.k = None  # Growth rate
        self.t0 = None  # Midpoint
        self.pre_fitted_bool = False
        self.fitted_bool = False

    @staticmethod
    def _logistic(X, k, t0):
        return 1 / (1 + np.exp(-k * (X - t0)))
    
    def learn_bounds(self, scenario: pd.Series):

        X_scenario = scenario.index.values
        Y_scenario = scenario.values
        
        # Provide bounds for parameters
        bounds = ([0, 2025], [1, 2050])

        # Provide initial guesses for k and t0
        p0 = [0.3, 2030]

        try:
            popt, _ = curve_fit(self._logistic, X_scenario, Y_scenario, p0=p0, bounds=bounds)
            self.k_init, self.t0_init = popt
        except RuntimeError:
            raise ValueError("Logistic model pre-fitting failed. Try different initial parameters or check your data.")
        
        self.pre_fitted_bool = True
        
        return self


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Fit logistic growth model to training data.
        X_train and Y_train should be 1D arrays.
        """
        if not self.pre_fitted_bool:
            raise RuntimeError("Model must be fitted before calling predict.")
        
        X_train = X_train.flatten()
        Y_train = Y_train.flatten()

        # Provide bounds for parameters
        k_lower = self.k_init - 0.05
        k_upper = self.k_init + 0.05
        t0_lower = self.t0_init - 5
        t0_upper = self.t0_init + 5
        bounds = ([k_lower, t0_lower], [k_upper, t0_upper])

        # Provide initial guesses for k and t0
        p0 = [self.k_init, self.t0_init]

        try:
            popt, _ = curve_fit(self._logistic, X_train, Y_train, p0=p0, bounds=bounds)
            self.k, self.t0 = popt
            self.fitted_bool = True
        except RuntimeError:
            raise ValueError("Logistic model fitting failed. Try different initial parameters or check your data.")

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted_bool:
            raise RuntimeError("Model must be fitted before calling predict.")

        return self._logistic(X_test.flatten(), self.k, self.t0)