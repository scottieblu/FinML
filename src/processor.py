from multiprocessing import cpu_count

import src.features.bars as bar
import src.features.datacleaner as dc
import src.features.indicators.bollinger_band
import src.features.indicators.crossover
import src.features.labeller as lab
from src.utils.utils import *


class FeatureProc:
    # Classes for data processing
    data_cleaner = dc.DataCleaner()
    bars = bar.Bars()
    labeller = lab.Labeller()
    crossover = src.features.indicators.crossover.MACrossover()

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

        if platform.system() == "Windows":
            self.cpus = 1
        else:
            self.cpus = cpu_count() - 1

    def loadParquetData(self, file_name):
        # Load from parquet
        infp = PurePath(self.data_raw_dir / file_name)
        return pd.read_parquet(infp)

    def cleanResample(self, df, frequency='1T'):
        df = data_cleaner.remove_outliers(df, 'price')
        return df.resample(frequency).median().drop_duplicates().dropna()

    def dollarBars(self, df):
        dbars = self.bars.dollar_bar_df(df, 'dv', 10_000).drop_duplicates().dropna()
        close = dbars.price.copy()
        return self.labeller.getDailyVol(close).dropna(), close

    def calcFeatures(self, file_name):
        self.df = self.loadParquetData(file_name)
        self.df = self.cleanResample(self.df, '1T')
        self.dailyVol, self.close = self.dollarBars(self.df)
        self.tEvents = labeller.getTEvents(self.close, h=self.dailyVol.mean())
        self.t1 = labeller.addVerticalBarrier(tEvents, self.close, numDays=1)

        # special features
        # moving average
        fast_window = 3
        slow_window = 7
        ma_side, close_df = self.crossover.get_side(self.close, fast_window, slow_window)

        # bollinger bands
        window = 20
        numsd = 1
        bband = src.features.indicators.bollinger_band.BollingerBand()
        bb_side_raw = bband.get_side(close, window=window, numsd=numsd)

        minRet = .01
        ptsl = [0, 2]
        cpus = cpu_count() - 1

        bb_events = labeller.getEvents(self.close, self.tEvents, ptsl, self.dailyVol, minRet, cpus, t1=self.t1,
                                       side=bb_side_raw)
        bb_side = bb_events.dropna().side

        features = (pd.DataFrame()
                    .assign(vol=bb_events.trgt)
                    .assign(ma_side=ma_side)
                    # .assign(srl_corr=srl_corr)
                    .drop_duplicates()
                    .dropna())
