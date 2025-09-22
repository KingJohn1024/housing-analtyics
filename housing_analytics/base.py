import os
import logging
import pandas as pd


class Base(object):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Set the logging level based on the verbose parameter
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.CRITICAL)

        # Set the shared project_path value
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.project_path = os.path.dirname(self.root_path)

    def load_df_sold(self) -> pd.DataFrame:
        fpath = f'{self.root_path}/data/df_sold.pq'
        df = pd.DataFrame()
        if os.path.exists(fpath):
            df = pd.read_parquet(f'{self.root_path}/data/df_sold.pq')
        return df

    def load_df_listing(self) -> pd.DataFrame:
        fpath = f'{self.root_path}/data/df_listing.pq'
        df = pd.DataFrame()
        if os.path.exists(fpath):
            df = pd.read_parquet(f'{self.root_path}/data/df_listing.pq')
        return df

    def get_all_zpids(self) -> list[str]:
        df_sold = self.load_df_sold()
        df_listing = self.load_df_listing()
        zpids = []

        if 'zpid' in df_sold.columns:
            zpids += df_sold.zpid.to_list()

        if 'zpid' in df_listing.columns:
            zpids += df_listing.zpid.to_list()

        return list(set(zpids))
