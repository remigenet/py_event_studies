import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Iterable, Callable
from functools import cached_property
from py_event_studies import config
# Import AR (Abnormal Return) test functions
from py_event_studies._ar_statistic_tests import (
    standard_test as ar_standard_test,
    bmp_test as ar_bmp_test,
    ordinary_cross_sec_test as ar_ordinary_cross_sec_test,
    kp_test as ar_kp_test, 
    create_avg_corrs
)

# Import CAR (Cumulative Abnormal Return) test functions
from py_event_studies._car_statistic_tests import (
    standard_test as car_standard_test,
    bmp_test as car_bmp_test,
    ordinary_cross_sec_test as car_ordinary_cross_sec_test,
    kp_test as car_kp_test
)

class Results:
    def __init__(self, date: str, ptf: Union[int, List[int]], 
                 estim_stock_returns: np.ndarray, event_stock_returns: np.ndarray,
                 estim_residuals: np.ndarray, event_residuals: np.ndarray,
                 event_d: np.ndarray):
        self.date = date
        self.ptf = [ptf] if isinstance(ptf, int) else ptf
        self.estim_stock_returns = estim_stock_returns
        self.event_stock_returns = event_stock_returns
        self.estim_residuals = estim_residuals
        self.event_residuals = event_residuals
        self.n_stocks = len(self.ptf)
        self.event_d = event_d
        
        self.cluster_num = config.cluster_num_list
        self.model_names = ['Cluster only', 'Cluster + Market', 'Cluster + FF3', 'Cluster + FF5',
                            'Market Model', 'FF3', 'FF5', 'Ridge in Cluster', 'Lasso in Cluster', 'ElasticNet in Cluster']
        self.test_names = ['std', 'CS', 'BMP', 'KP']
        self.models_degree_of_freedom = np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1
        self.n_stocks = len(self.ptf)

        self.event_idx = int(len(event_residuals)/2) + 1
        self.avg_cors = create_avg_corrs(estim_residuals)


    @cached_property
    def std_test_result(self):
        return ar_standard_test(self.event_residuals[:, : , self.event_idx, :], self.estim_residuals) 
    
    @cached_property
    def bmp_test_result(self):
        return ar_bmp_test(self.event_residuals[:, : , self.event_idx, :], self.estim_residuals, self.event_d[:, : , self.event_idx, :])

    @cached_property
    def ocs_test_result(self):
        return ar_ordinary_cross_sec_test(self.event_residuals[:, : , self.event_idx, :])

    @cached_property
    def kp_test_results(self):
        return ar_kp_test(self.event_residuals[:, : , self.event_idx, :], self.estim_residuals, self.avg_cors)

    @cached_property
    def std_test_cumulative_result(self):
        return car_standard_test(self.event_residuals, self.estim_residuals) 
    
    @cached_property
    def bmp_test_cumulative_result(self):
        return car_bmp_test(self.event_residuals, self.estim_residuals, self.event_d)

    @cached_property
    def ocs_test_cumulative_result(self):
        return car_ordinary_cross_sec_test(self.event_residuals)

    @cached_property
    def kp_test_cumulative_results(self):
        return car_kp_test(self.event_residuals, self.estim_residuals, self.avg_cors)

    @cached_property
    def estim_preds(self):
        return self.estim_stock_returns + self.estim_residuals

    @cached_property
    def event_preds(self):
        return self.event_stock_returns + self.event_residuals

    @cached_property
    def std_test_stats(self):
        return pd.DataFrame(self.std_test_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def cs_test_stats(self):
        return pd.DataFrame(self.ocs_test_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def bmp_test_stats(self):
        return pd.DataFrame(self.bmp_test_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def kp_test_stats(self):
        return pd.DataFrame(self.kp_test_results, index=self.cluster_num, columns=self.model_names)
    
    @cached_property
    def std_cumulative_test_stats(self):
        return pd.DataFrame(self.std_test_cumulative_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def cs_cumulative_test_stats(self):
        return pd.DataFrame(self.ocs_test_cumulative_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def bmp_cumulative_test_stats(self):
        return pd.DataFrame(self.bmp_test_cumulative_result, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def kp_cumulative_test_stats(self):
        return pd.DataFrame(self.kp_test_cumulative_results, index=self.cluster_num, columns=self.model_names)

    @staticmethod
    def _t_distribution_cdf(x, df):
        """
        Approximation of the CDF of the t-distribution.
        This is a simple approximation and might not be as accurate as scipy's implementation for all cases.
        """
        return 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-0.5 * x**2 * (df + 1) / (x**2 + df)))

    def _calculate_p_values(self, test_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for given test statistics."""
        df = self.n_stocks - self.models_degree_of_freedom
        p_values = 2 * (1 - self._t_distribution_cdf(np.abs(test_stats), df=df[np.newaxis, :]))
        return pd.DataFrame(p_values, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def std_p_values(self):
        return self._calculate_p_values(self.std_test_stats)

    @cached_property
    def cs_p_values(self):
        return self._calculate_p_values(self.cs_test_stats)

    @cached_property
    def bmp_p_values(self):
        return self._calculate_p_values(self.bmp_test_stats)

    @cached_property
    def kp_p_values(self):
        return self._calculate_p_values(self.kp_test_stats)

    @cached_property
    def std_cumulative_p_values(self):
        return self._calculate_p_values(self.std_cumulative_test_stats)

    @cached_property
    def cs_cumulative_p_values(self):
        return self._calculate_p_values(self.cs_cumulative_test_stats)

    @cached_property
    def bmp_cumulative_p_values(self):
        return self._calculate_p_values(self.bmp_cumulative_test_stats)

    @cached_property
    def kp_cumulative_p_values(self):
        return self._calculate_p_values(self.kp_cumulative_test_stats)

    def get_test_result(self, test_name: str, cumulative: bool = False) -> pd.DataFrame:
        """Get the results for a specific test as a DataFrame."""
        if test_name not in self.test_names:
            raise ValueError(f"Invalid test name. Choose from {self.test_names}")
        
        return getattr(self, f"{test_name.lower()}{'_cumulative' if cumulative else''}_test_stats")

    def get_p_values(self, test_name: str, cumulative: bool = False) -> pd.DataFrame:
        """Get p-values for a specific test."""
        if test_name not in self.test_names:
            raise ValueError(f"Invalid test name. Choose from {self.test_names}")
        
        return getattr(self, f"{test_name.lower()}{'_cumulative' if cumulative else''}_p_values")

    def summary(self) -> None:
        try:
            from IPython.display import display
            print = display
        except ImportError:
            pass
        """Print a summary of the results."""
        print(f"Event Date: {self.date}")
        print(f"Portfolio: {self.ptf}")
        print(f"Number of stocks: {self.n_stocks}")
        print(f"Estimation period: {self.estim_stock_returns.shape[2]} days")
        print(f"Event period: {self.event_stock_returns.shape[2]} days")
        print("\nTest Results:")
        for test in self.test_names:
            print(f"\nNon Cumulative {test} Test:")
            print(self.get_test_result(test))
            print(f"\nNon Cumulative{test} Test P-values:")
            print(self.get_p_values(test))
            print(f"\Cumulative {test} Test:")
            print(self.get_test_result(test, cumulative=True))
            print(f"\Cumulative{test} Test P-values:")
            print(self.get_p_values(test, cumulative=True))

    def __str__(self):
        return f"Event Study Results\n" \
               f"Event Date: {self.date}\n" \
               f"Portfolio: {self.ptf[:5]}{'...' if len(self.ptf) > 5 else ''}\n" \
               f"Number of stocks: {self.n_stocks}\n" \
               f"Estimation period: {self.estim_stock_returns.shape[2]} days\n" \
               f"Event period: {self.event_stock_returns.shape[2]} days\n" \
               f"Number of cluster configurations: {len(self.cluster_num)}\n" \
               f"Number of models: {len(self.model_names)}\n" \
               f"Available tests: {', '.join(self.test_names)}"

    @staticmethod
    def help():
        print("Available methods and properties:")
        print("- estim_preds: Predicted returns for estimation period")
        print("- event_preds: Predicted returns for event period")
        print("- std_test_stats, cs_test_stats, bmp_test_stats, kp_test_stats: Test statistics")
        print("- std_p_values, cs_p_values, bmp_p_values, kp_p_values: P-values for tests")
        print("- get_test_result(test_name): Get test statistics for a specific test")
        print("- get_p_values(test_name): Get p-values for a specific test")
        print("- summary(): Print a detailed summary of results")
        print("- plot(cluster_idx, model_idx): Plot true vs predicted returns for a specific model and cluster configuration")
        print("- to_excel(filename): Export results to an Excel file")

    def plot(self, cluster_num: int, model_name: str, only_event=False):
        """
        Plot true vs predicted returns for a specific model and cluster configuration.
        
        Args:
        cluster_num (int): Number of clusters to plot
        model_name (str): Name of the model to plot
        """
        if cluster_num not in self.cluster_num:
            raise ValueError(f"cluster_num must be one of {self.cluster_num}")
        if model_name not in self.model_names:
            raise ValueError(f"model_name must be one of {self.model_names}")

        cluster_idx = self.cluster_num.index(cluster_num)
        model_idx = self.model_names.index(model_name)

        estim_true = self.estim_stock_returns[cluster_idx, model_idx]
        estim_pred = self.estim_preds[cluster_idx, model_idx]
        event_true = self.event_stock_returns[cluster_idx, model_idx]
        event_pred = self.event_preds[cluster_idx, model_idx]

        n_stocks = self.n_stocks
        n_rows = (n_stocks + 3) // 4  # Round up to nearest multiple of 4

        fig, axs = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
        fig.suptitle(f"True vs Predicted Returns - Cluster: {cluster_num}, Model: {model_name}")

        for i in range(n_stocks):
            row = i // 4
            col = i % 4
            ax = axs[row, col] if n_rows > 1 else axs[col]

            # Plot estimation period
            if not only_event:
                ax.plot(range(len(estim_true)), estim_true[:, i], label='True (Estimation)', color='blue', alpha=0.7)
                ax.plot(range(len(estim_pred)), estim_pred[:, i], label='Predicted (Estimation)', color='red', alpha=0.7)

            # Plot event period
            if not only_event:
                offset = len(estim_true)
            else:
                offset = 0
            ax.plot(range(offset, offset + len(event_true)), event_true[:, i], label='True (Event)', color='green', alpha=0.7)
            ax.plot(range(offset, offset + len(event_pred)), event_pred[:, i], label='Predicted (Event)', color='orange', alpha=0.7)

            ax.set_title(f"Stock {self.ptf[i]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Returns")
            ax.legend(loc='upper left', fontsize='x-small')

        # Remove any unused subplots
        for i in range(n_stocks, n_rows * 4):
            row = i // 4
            col = i % 4
            fig.delaxes(axs[row, col] if n_rows > 1 else axs[col])

        plt.tight_layout()
        plt.show()

    def to_excel(self, filename: str):
        if importlib.util.find_spec("openpyxl") is None:
            raise ImportError("openpyxl is required for Excel export. "
                              "Install it with 'pip install openpyxl' or "
                              "'poetry install --with excel'")
        
        with pd.ExcelWriter(filename) as writer:
            for test in self.test_names:
                self.get_test_result(test).to_excel(writer, sheet_name=f"{test} Test")
                self.get_p_values(test).to_excel(writer, sheet_name=f"{test} P-values")
            
            for i, cluster_num in enumerate(self.cluster_num):
                for j, model in enumerate(self.model_names):
                    # Estimation period
                    sheet_name = f"Estim {cluster_num}_{model}"
                    estim_df = pd.DataFrame(self.estim_residuals[i, j], columns=[f"{ptf}_residual" for ptf in self.ptf])
                    estim_df = pd.concat([
                        estim_df,
                        pd.DataFrame(self.estim_stock_returns[i, j], columns=[f"{ptf}_return" for ptf in self.ptf])
                    ], axis=1)
                    estim_df.to_excel(writer, sheet_name=sheet_name)
                    
                    # Event period
                    sheet_name = f"Event {cluster_num}_{model}"
                    event_df = pd.DataFrame(self.event_residuals[i, j], columns=[f"{ptf}_residual" for ptf in self.ptf])
                    event_df = pd.concat([
                        event_df,
                        pd.DataFrame(self.event_stock_returns[i, j], columns=[f"{ptf}_return" for ptf in self.ptf])
                    ], axis=1)
                    event_df.to_excel(writer, sheet_name=sheet_name)

        print(f"Results saved to {filename}")


    def create_shocked_results(self, shock: Union[np.ndarray, Iterable[float], float], c_vector_generator: Callable = lambda shape: np.random.uniform(1, 2, shape)):
        """
        Create shocked results based on the given shock and c_vector_generator.

        Args:
            shock (Union[np.ndarray, Iterable[float], float]): The shock to apply. If an iterable, it should have the same length as the event period. Otherwise, it will be applied only to event date.
            c_vector_generator (Callable): A function that takes a shape and returns a vector of that shape.

        Returns:
            Results: A new Results object with shocked event residuals.
        """
        c_vector = c_vector_generator((self.event_residuals.shape[-1],))
        c_vector = np.expand_dims(c_vector, axis=(0, 1))
        c_vector_broadcasted = np.broadcast_to(c_vector, (*self.event_residuals.shape[:2], *self.event_residuals.shape[3:]))
        shocked_residuals = self.event_residuals.copy()
        shocked_event_returns = self.event_stock_returns.copy()

        if isinstance(shock, (np.ndarray, Iterable)) and not isinstance(shock, str):
            shock_array = np.array(shock, dtype=float)
            if shock_array.shape != (self.event_residuals.shape[2],):
                raise ValueError(f"The shock iterable must have {self.event_residuals.shape[2]} elements, equal to the event period length.")
            
            # Reshape shock_array to allow broadcasting
            shock_array = shock_array.reshape((1, 1, -1, 1))
            
            # Apply shock to all days
            shocked_residuals = shocked_residuals * c_vector_broadcasted + shock_array
        else:
            # If shock is a float, apply it only at the event day
            shocked_residuals[:, :, self.event_idx, :] = shocked_residuals[:, :, self.event_idx, :] * c_vector_broadcasted + shock

        shocked_event_returns = self.event_preds - shocked_residuals

        return Results(
            date=self.date,
            ptf=self.ptf,
            estim_stock_returns=self.estim_stock_returns,
            event_stock_returns=shocked_event_returns,
            estim_residuals=self.estim_residuals,
            event_residuals=shocked_residuals,
            event_d=self.event_d
        )

        