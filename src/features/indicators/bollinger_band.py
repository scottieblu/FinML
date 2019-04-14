import numpy as np
import pandas as pd


class BollingerBand:

    def bbands(self, price, window, width, numsd):
        """
        returns average, upper band, and lower band
        """
        ave = price.rolling(window).mean()
        sd = price.rolling(window).std(ddof=0)
        if width:
            upband = ave * (1 + width)
            dnband = ave * (1 - width)
            return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)
        if numsd:
            upband = ave + (sd * numsd)
            dnband = ave - (sd * numsd)
            return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)

    def get_up_cross(self, df, col):
        # col is price column
        crit1 = df[col].shift(1) < df.upper.shift(1)
        crit2 = df[col] > df.upper
        return df[col][crit1 & crit2]

    def get_down_cross(self, df, col):
        # col is price column
        crit1 = df[col].shift(1) > df.lower.shift(1)
        crit2 = df[col] < df.lower
        return df[col][crit1 & crit2]

    def get_side(self, price, window=None, width=None, numsd=None):
        bb_df = pd.DataFrame()
        bb_df['price'], bb_df['ave'], bb_df['upper'], bb_df['lower'] = self.bbands(price, window, width, numsd)
        bb_df.dropna(inplace=True)
        bb_up = self.get_up_cross(bb_df, 'price')
        bb_down = self.get_down_cross(bb_df, 'price')

        bb_side_up = pd.Series(-1, index=bb_up.index)  # sell on up cross for mean reversion
        bb_side_down = pd.Series(1, index=bb_down.index)  # buy on down cross for mean reversion
        return pd.concat([bb_side_up, bb_side_down]).sort_index()
