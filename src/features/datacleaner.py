import numpy as np
from numba import jit


class DataCleaner:

    @staticmethod
    @jit(nopython=True)
    def mad_outlier(y, thresh=3.):
        """
        compute outliers based on mad
        # args
            y: assumed to be array with shape (N,1)
            thresh: float()
        # returns
            array index of outliers
        """
        median = np.median(y)
        diff = np.sum((y - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def remove_outliers(self, df, df_column):
        mad = self.mad_outlier(df[df_column].to_numpy().reshape(-1, 1))
        return df.loc[~mad]
