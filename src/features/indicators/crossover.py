import pandas as pd


class MACrossover:
    """
      Moving average crossover strategy
    """

    def close_df(self, close, fast_window, slow_window):
        return (pd.DataFrame()
                .assign(price=close)
                .assign(fast=close.ewm(fast_window).mean())
                .assign(slow=close.ewm(slow_window).mean()))

    def get_up_cross(self, df):
        crit1 = df.fast.shift(1) < df.slow.shift(1)
        crit2 = df.fast > df.slow
        return df.fast[(crit1) & (crit2)]

    def get_down_cross(self, df):
        crit1 = df.fast.shift(1) > df.slow.shift(1)
        crit2 = df.fast < df.slow
        return df.fast[(crit1) & (crit2)]

    def get_side(self, close, fast_window=3, slow_window=7):
        """
        Return the meta labels for the stragegy.
        +1 strategy predicts an expected upswing event at time t
        -1 strategy predicts an expected downswing event at time t
        :param close:
        :param fast_window:
        :param slow_window:
        :return:
        """
        close_df = self.close_df(close, fast_window, slow_window)
        up = self.get_up_cross(close_df)
        down = self.get_down_cross(close_df)
        return pd.concat([pd.Series(1, index=up.index), pd.Series(-1, index=down.index)]).sort_index()
