import platform
from multiprocessing import cpu_count
from pathlib import PurePath

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import src.preprocessing.bars as bar
import src.preprocessing.datacleaner as dc
import src.preprocessing.features.bollinger_band
import src.preprocessing.features.crossover
import src.preprocessing.labeller as lab
from src.utils.utils import *


class Processor:
    # Classes for data processing
    data_cleaner = dc.DataCleaner()
    bars = bar.Bars()
    labeller = lab.Labeller()
    crossover = src.preprocessing.features.crossover.MACrossover()

    def __init__(self, pdir):
        self.data_dir = pdir / 'data'
        self.data_raw_dir = self.data_dir / 'raw'
        self.data_interim_dir = self.data_dir / 'interim'
        self.data_processed_dir = self.data_dir / 'processed'

        # Variables for the data
        self.df = []
        self.dailyVol = []
        self.tEvents = []
        self.t1 = []
        self.close = []
        self.features = []
        self.bins = []

        if platform.system() == "Windows":
            self.cpus = 1
        else:
            self.cpus = cpu_count() - 1

    def loadParquetData(self, file_name):
        # Load from parquet
        infp = PurePath(self.data_raw_dir / file_name)
        return pd.read_parquet(infp)

    def cleanResample(self, df, frequency='1T'):
        df = self.data_cleaner.remove_outliers(df, 'price')
        return df.resample(frequency).median().drop_duplicates().dropna()

    def dollarBars(self, df):
        dbars = self.bars.dollar_bar_df(df, 'dv', 10_000).drop_duplicates().dropna()
        close = dbars.price.copy()
        return self.labeller.getDailyVol(close).dropna(), close

    def returns(self, s):
        arr = np.diff(np.log(s))
        return (pd.Series(arr, index=s.index[1:]))

    def df_rolling_autocorr(self, df, window, lag=1):
        """Compute rolling column-wise autocorrelation for a DataFrame."""

        return (df.rolling(window=window)
                .corr(df.shift(lag)))  # could .dropna() here

    def calcFeatures(self, file_name):
        self.df = self.loadParquetData(file_name)
        self.df = self.cleanResample(self.df, '1T')
        self.dailyVol, self.close = self.dollarBars(self.df)
        self.tEvents = self.labeller.getTEvents(self.close, h=self.dailyVol.mean())
        self.t1 = self.labeller.addVerticalBarrier(self.tEvents, self.close, numDays=1)

        # special features
        # moving average
        fast_window = 3
        slow_window = 7
        ma_side, close_df = self.crossover.get_side(self.close, fast_window, slow_window)

        # bollinger bands
        window = 20
        numsd = 1
        bband = src.preprocessing.features.bollinger_band.BollingerBand()
        bb_side_raw = bband.get_side(self.close, window=window, numsd=numsd)

        minRet = .01
        ptsl = [0, 2]
        cpus = cpu_count() - 1

        bb_events = self.labeller.getEvents(self.close, self.tEvents, ptsl, self.dailyVol, minRet, cpus, t1=self.t1,
                                            side=bb_side_raw)
        bb_side = bb_events.dropna().side

        srl_corr = self.df_rolling_autocorr(self.returns(self.close), window=window).rename('srl_corr')

        features = (pd.DataFrame()
                    .assign(vol=bb_events.trgt)
                    .assign(ma_side=ma_side)
                    .assign(srl_corr=srl_corr)
                    .drop_duplicates()
                    .dropna())

        bb_bins = self.labeller.getBins(bb_events, self.close).dropna()

        return features, bb_bins

    def classifier(self, features, bins):
        Xy = (pd.merge_asof(features, bins[['bin']],
                            left_index=True, right_index=True,
                            direction='forward').dropna())

        X = Xy.drop('bin', axis=1).values
        y = Xy['bin'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            shuffle=False)

        n_estimator = 10000
        rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                                    criterion='entropy')
        rf.fit(X_train, y_train)

        # The random forest model by itself
        y_pred_rf = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)

        return y_pred, y_pred_rf

    def process(self, file_name):
        self.features, self.bins = self.calcFeatures(file_name)
        y_pred, y_pred_rf = self.classifier(self.features, self.bins)
