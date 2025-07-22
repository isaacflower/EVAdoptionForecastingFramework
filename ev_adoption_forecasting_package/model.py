# Import modules
from ev_adoption_forecasting_package.mean_function import CustomMeanFunction, Spline
from ev_adoption_forecasting_package.transforms import probit, invprobit

# Data
import pandas as pd
import geopandas as gpd
import numpy as np
seed = 42
rng = np.random.default_rng(seed)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("paper")

# Gaussian Processes
import gpflow
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_summary_fmt("notebook")
f64 = gpflow.utilities.to_default_float # convert to float64 for tfp to play nicely with gpflow

class GPForecastingModel:
    def __init__(
            self, 
        ):
        self.area_id: str = None
        self.data: pd.Series = None
        self.probit_data: pd.Series = None
        self.group_scenario: pd.Series = None
        self.training_data: tuple[np.ndarray, np.ndarray] = None
        self.testing_data: tuple[np.ndarray, np.ndarray] = None
        self.t_0: int = None
        self.spl: Spline = None
        self.mean_function: CustomMeanFunction = None
        self.h_f: int = None
        self.t_n: int = None
        self.t_f: int = None
        pass
    
    # Core Methods

    def load_data(self, data: pd.Series, area_id: str, group_data: pd.Series, future_group_data: dict):
        self.area_id = area_id
        self.data = data.copy()
        masked = self._mask_zeros(self.data)
        self.probit_data = masked.apply(probit)
        self.group_scenario = self._create_group_scenario(group_data, future_group_data)
        self.t_0 = self.group_scenario.index[0]
        self.mean_function = self._prepare_mean_function()
        
    def prepare_train_test_data(self, t_dict: dict):
        self.t_n = t_dict['t_n']
        self.t_f = t_dict['t_f']
        self.h_f = self.t_f - self.t_n
        X_train, Y_train = self._prepare_training_data(h_f=self.h_f, t_0=self.t_0, training_data=self.probit_data)
        X_test, Y_test = self._prepare_testing_data(h_f=self.h_f, t_0=self.t_0, training_data=self.probit_data)

        self.training_data = (X_train, Y_train)
        self.testing_data = (X_test, Y_test)
        pass
    
    def build_gp_model(
            self,
            kernel: gpflow.kernels.Kernel, 
            likelihood: gpflow.likelihoods.Likelihood,
        ) -> gpflow.models.GPR: 

        if self.training_data is None:
            raise ValueError("Training data not prepared. Please call prepare_train_test_data() first.")

        X_train, Y_train = self.training_data

        gp_model = gpflow.models.GPR(
            (f64(X_train), f64(Y_train)),
            kernel=kernel,
            mean_function=self.mean_function,
            likelihood=likelihood
        )
        
        self.gp_model = gp_model
        return self.gp_model
    
    def make_forecast(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_new = X_new.astype(np.float64)  # Ensure float64
        X_new = X_new - float(self.t_0)  # Ensure consistent type

        if np.any(X_new < 0):
            raise ValueError("X_new is less than t_0 which is not allowed.")
        
        if self.gp_model is None:
            raise ValueError("GP Model not built. Please call build_gp_model() first.")
        
        y_mean, y_var = self.gp_model.predict_y(X_new) # Using y instead of f to include observation noise
        return y_mean.numpy(), y_var.numpy()
    
    def generate_sample_forecasts(self, X_new: np.ndarray, num_samples: int = 10) -> np.ndarray:
        X_new = X_new.astype(np.float64)  # Ensure float64
        X_new = X_new - float(self.t_0)  # Ensure consistent type
        f_samples = self.gp_model.predict_f_samples(X_new, num_samples)[:, :, 0].numpy().T
        return f_samples
    
    def plot_forecast(
            self,
            X_plot: np.ndarray,
            X_new: np.ndarray,
            probit_transform: bool
        ):
        X_train, Y_train = self.training_data
        X_test, Y_test = self.testing_data
        f_mean, f_var = self.make_forecast(X_new=X_plot)
        f_mean_new, _ = self.make_forecast(X_new=X_new)
        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
        mean_function = probit(self.spl.evaluate(X_plot - self.t_0))

        if not probit_transform:
            Y_train = invprobit(Y_train)
            Y_test = invprobit(Y_test)
            f_mean = invprobit(f_mean)
            f_mean_new = invprobit(f_mean_new)
            f_lower = invprobit(f_lower)
            f_upper = invprobit(f_upper)
            mean_function = invprobit(mean_function)

        plt.figure(figsize=(6, 4))

        plt.plot(X_train + self.t_0, Y_train, "x", mew=1, label="Training data", color='black', zorder=10, clip_on=False)
        plt.plot(X_test + self.t_0, Y_test, "x", mew=1, label="Test data", color='C1', zorder=10, clip_on=False)
        plt.plot(X_new, f_mean_new, "--", color="C2", label="GP Mean (Forecast)", zorder=10)
        if self.h_f == 0:
            plt.plot(X_plot, f_mean, "-", color="C0", label="GP Mean (Historical)")
        else:
            plt.plot(X_plot[:-(self.t_f - self.t_n)], f_mean[:-(self.t_f - self.t_n)], "-", color="C0", label="GP Mean (Historical)")
        plt.plot(X_plot, mean_function, color='grey', linestyle="--", label="Mean Function", zorder=10)

        plt.fill_between(X_plot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.2, label="95% CI")
        
        plt.xlim(X_plot[0], X_plot[-1])
        plt.xlabel('Year')

        if probit_transform:
            plt.ylabel('EV Market Share (Probit Transformed)')
        
        elif not probit_transform:
            plt.ylim(bottom=0)
            plt.ylabel('EV Market Share')
        
        plt.legend()
        plt.show()

    # Convenience Functions

    def _create_group_scenario(self, group_data: pd.Series, future_group_data: dict) -> pd.Series:
        return pd.concat([group_data, pd.Series(future_group_data)])
    
    def _prepare_mean_function(self):
        self.spl = self._fit_spline(self.group_scenario)
        self.mean_function = CustomMeanFunction(mean_function=self.spl)
        return self.mean_function

    def _fit_spline(self, group_scenario: pd.Series) -> Spline:
        x = group_scenario.index - self.t_0
        y = group_scenario.values
        spl = Spline(x, y)
        spl.fit_spline()
        return spl
    
    def _mask_zeros(self, series: pd.Series) -> pd.Series:
        series = series.copy()
        zero_indices = series.index[series == 0]
        if not zero_indices.empty:
            last_zero_idx = zero_indices[-1]
            series.loc[:last_zero_idx] = np.nan
        return series
    
    def _prepare_training_data(self, h_f: int, t_0: int, training_data: pd.Series):
        if h_f == 0:
            X_train = training_data.index.values - t_0
            Y_train = training_data.values
        else:
            X_train = training_data.index.values[:-h_f] - t_0
            Y_train = training_data.values[:-h_f]
        
        len_X_train_raw = len(X_train)
        
        X_train = X_train[~np.isnan(Y_train)]
        Y_train = Y_train[~np.isnan(Y_train)]

        if np.isnan(Y_train).all():
            X_train = np.arange(len_X_train_raw - 1, len_X_train_raw)
            Y_train = probit(np.zeros(len(X_train)))

        return X_train.reshape(-1, 1), Y_train.reshape(-1, 1)

    def _prepare_testing_data(self, h_f: int, t_0: int, training_data: pd.Series):
        if self.h_f == 0:
            X_test = np.array([])
            Y_test = np.array([])
        else:
            X_test = training_data.index.values[-h_f:] - t_0
            Y_test = training_data.values[-h_f:]
            Y_test[np.isnan(Y_test)] = probit(0)
        return X_test.reshape(-1, 1), Y_test.reshape(-1, 1)