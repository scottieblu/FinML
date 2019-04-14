import matplotlib.pyplot as plt
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

    def get_ohlc(self, ref, sub):
        """
        fn: get ohlc from custom bars

        # args
            ref : reference pandas series with all prices
            sub : custom tick pandas series
        # returns
            tick_df : dataframe with ohlc values
        """
        ohlc = []
        for i in tqdm(range(sub.index.shape[0] - 1)):
            start, end = sub.index[i], sub.index[i + 1]
            tmp_ref = ref.loc[start:end]
            max_px, min_px = tmp_ref.max(), tmp_ref.min()
            o, h, l, c = sub.iloc[i], max_px, min_px, sub.iloc[i + 1]
            ohlc.append((end, start, o, h, l, c))
        cols = ['end', 'start', 'open', 'high', 'low', 'close']
        return pd.DataFrame(ohlc, columns=cols)

    def count_bars(self, df, group='1W', price_col='price'):
        return df.groupby(pd.TimeGrouper(group))[price_col].count()

    def scale(self, s):
        return (s - s.min()) / (s.max() - s.min())

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

    def select_sample_data(self, ref, sub, col, date):
        """
        select a sample of data based on date, assumes datetimeindex

        # args
            ref: pd.DataFrame containing all ticks
            sub: subordinated pd.DataFrame of prices
            col: str(), column to use
            date: str(), date to select
        # returns
            xdf: ref pd.Series
            xtdf: subordinated pd.Series
        """
        xdf = ref[col].loc[date]
        xtdf = sub[col].loc[date]
        return xdf, xtdf

    def plot_sample_data(self, ref, sub, bar_type, *args, **kwds):
        f, axes = plt.subplots(3, sharex='all', sharey='all', figsize=(10, 7))
        ref.plot(*args, **kwds, ax=axes[0], label='price')
        sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
        axes[0].legend()

        ref.plot(*args, **kwds, ax=axes[1], label='price', marker='o')
        sub.plot(*args, **kwds, ax=axes[2], ls='', marker='X',
                 color='r', label=bar_type)

        for ax in axes[1:]: ax.legend()
        plt.tight_layout()
