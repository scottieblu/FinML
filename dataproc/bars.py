import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm


class Bars:
    """
      Class to calculate different types of price based bars.
    """

    def returns(self, s):
        arr = np.diff(np.log(s))
        return pd.Series(arr, index=s.index[1:])

    def tick_bars(self, df, price_column, m):
        """
        compute tick bars
        # args
            df: pd.DataFrame()
            column: name for price dataproc
            m: int(), threshold value for ticks
        # returns
            idx: list of indices
        """
        t = df[price_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += 1
            if ts >= m:
                idx.append(i)
                ts = 0
                continue
        return idx

    def tick_bar_df(self, df, price_column, m):
        idx = self.tick_bars(df, price_column, m)
        return df.iloc[idx]

    def volume_bars(self, df, volume_column, m):
        """
        compute volume bars
        # args
            df: pd.DataFrame()
            column: name for volume dataproc
            m: int(), threshold value for volume
        # returns
            idx: list of indices
        """
        t = df[volume_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += x
            if ts >= m:
                idx.append(i)
                ts = 0
                continue
        return idx

    def volume_bar_df(self, df, volume_column, m):
        idx = self.volume_bars(df, volume_column, m)
        return df.iloc[idx]

    def dollar_bars(self, df, dv_column, m):
        """
        compute dollar bars
        # args
            df: pd.DataFrame()
            column: name for dollar volume dataproc
            m: int(), threshold value for dollars
        # returns
            idx: list of indices
        """
        t = df[dv_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += x
            if ts >= m:
                idx.append(i)
                ts = 0
                continue
        return idx

    def dollar_bar_df(self, df, dv_column, m):
        idx = self.dollar_bars(df, dv_column, m)
        return df.iloc[idx]

    @jit(nopython=True)
    def numba_is_close(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return np.fabs(a - b) <= np.fmax(rel_tol * np.fmax(np.fabs(a), np.fabs(b)), abs_tol)

    @jit(nopython=True)
    def bt(self, p0, p1, bs):
        # if math.is_close((p1 - p0), 0.0, abs_tol=0.001):
        if self.numba_is_close((p1 - p0), 0.0, abs_tol=0.001):
            b = bs[-1]
            return b
        else:
            b = np.abs(p1 - p0) / (p1 - p0)
            return b

    @jit(nopython=True)
    def get_imbalance(self, t):
        bs = np.zeros_like(t)
        for i in np.arange(1, bs.shape[0]):
            t_bt = self.bt(t[i - 1], t[i], bs[:i - 1])
            bs[i - 1] = t_bt
        return bs[:-1]  # remove last value
