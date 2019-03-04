import pandas as pd


class Reader:

    def read_kibot_ticks(self, fp):
        # read tick dataproc from http://www.kibot.com/support.aspx#data_format
        cols = list(map(str.lower, ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Size']))
        df = (pd.read_csv(fp, header=None)
              .rename(columns=dict(zip(range(len(cols)), cols)))
              .assign(dates=lambda df: (pd.to_datetime(df['date'] + df['time'],
                                                       format='%m/%d/%Y%H:%M:%S')))
              .assign(v=lambda df: df['size'])  # volume
              .assign(dv=lambda df: df['price'] * df['size'])  # dollar volume
              .drop(['date', 'time'], axis=1)
              .set_index('dates')
              .drop_duplicates())
        return df
