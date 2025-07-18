import pandas as pd
import numpy as np
import os
from typing import Optional

class LSOAVehicleRegistrationDataProcessor:
    def __init__(self, lsoa_lookup_path: str):
        self.v_reg_df_raw = None
        self.v_reg_df = None
        self.ev_reg_df_raw = None
        self.ev_reg_private_df = None
        self.ev_reg_company_df = None
        self.ev_reg_total_df = None
        self.ev_reg_df = None
        self.lsoa_lookup = pd.read_csv(lsoa_lookup_path)
        self.data_dict = None

    # === Core Methods ===
    
    def load_data(self, raw_data_path: str, meta_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw vehicle and EV registration data from disk based on metadata describing file structure."""
        for vehicle_type, details in meta_data.items():
            file_path = os.path.join(raw_data_path, details['file_name'])
            print(f"Loading {vehicle_type} data from {file_path}")
            if vehicle_type == 'v':
                self.v_reg_df_raw = self._load_csv(file_path, details)
            elif vehicle_type == 'ev':
                self.ev_reg_df_raw = self._load_csv(file_path, details)
        return self.v_reg_df_raw, self.ev_reg_df_raw
    
    def filter_data(self, filters_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filters raw registration datasets using custom query/filter instructions and prepares them for processing."""
        if self.v_reg_df_raw is None or self.ev_reg_df_raw is None:
            raise ValueError('Raw data not loaded. Please load raw data first.')

        vehicle_map = {
            'v': (self.v_reg_df_raw, 'v_reg_df'),
            'ev_private': (self.ev_reg_df_raw, 'ev_reg_private_df'),
            'ev_company': (self.ev_reg_df_raw, 'ev_reg_company_df'),
            'ev_total': (self.ev_reg_df_raw, 'ev_reg_total_df'),
        }

        for vehicle_type, filters in filters_dict.items():
            if vehicle_type not in vehicle_map:
                continue
            source_df, target_attr = vehicle_map[vehicle_type]
            filtered = self._filter_dataframe(source_df, filters)
            cleaned = filtered[~filtered.index.duplicated(keep='first')].T.iloc[::-1].astype(float)
            setattr(self, target_attr, cleaned)

        return self.v_reg_df, self.ev_reg_private_df, self.ev_reg_company_df, self.ev_reg_total_df
    
    def process_data(self, t_0: int, t_n: int) -> None:
        """Processes the filtered datasets: aligns structures, fills missing data, interpolates values, and prepares annualised outputs."""
        self._align_ev_reg_dfs_to_v_reg()
        self._fill_missing_private_ev_data()
        self._interpolate_missing_data()
        self.data_dict = {
            'v': self.v_reg_df,
            'ev': self.ev_reg_df,
        }
        self.data_dict = {k: df.set_index(df.index.map(self._convert_quarter_index_to_float)) for k, df in self.data_dict.items()}
        self.annual_data_dict = {k: self._annualise_data_last(df).loc[t_0:t_n] for k, df in self.data_dict.items()}
        self.annual_data_dict['ev_ms'] = self.annual_data_dict['ev'] / self.annual_data_dict['v']

    def filter_by_lads(self, lad_list: list):
        """Extracts data for specific Local Authority Districts (LADs) using the LSOA lookup table."""
        lad_lsoa_dict = {}
        for lad in lad_list:
            lad_lsoa_sub_dict = {}
            for key in self.annual_data_dict.keys():
                dict = self._filter_by_lad(LAD=lad)
                lad_lsoa_sub_dict[key] = dict[key]
            lad_lsoa_dict[lad] = lad_lsoa_sub_dict
        return lad_lsoa_dict

    def save_data(self, save_path: str, year_quarter: str):
        """Saves the processed and annualised datasets to disk, organized by type and quarter."""
        for name, df in self.annual_data_dict.items():
            df.index.name = 'Date'
            df.to_csv(os.path.join(save_path, f'{name}_{year_quarter}.csv'))
        print('Data saved successfully')

    # === Internal Helper Methods === 
    
    def _load_csv(self, filepath: str, details: dict) -> pd.DataFrame:
        csv_df = pd.read_csv(
            filepath, 
            dtype=self._apply_dtypes(details['first'], details['last']),
            na_values=details['na_values']
        )
        return csv_df

    def _apply_dtypes(self, first: int, last: int) -> dict:
        dtypes = {i: str for i in range(first)}
        dtypes.update({i: float for i in range(first, last)})
        return dtypes
    
    def _filter_dataframe(self, df_raw: pd.DataFrame, filters: dict) -> pd.DataFrame:
        df = (
            df_raw
            .query(filters['query'])
            .drop(filters['dropped_cols'], axis=1)
            .set_index('LSOA11CD')
        )
        return df
    
    def _align_ev_reg_dfs_to_v_reg(self):
        """
        Aligns ev_reg_df to have the same LSOA11CD indices as v_reg_df.
        Missing LSOAs will have NaN rows inserted.
        """
        target_index = self.v_reg_df.columns if isinstance(self.v_reg_df.columns, pd.Index) else self.v_reg_df.index

        def reindex_df(df):
            if df.index.name != "LSOA11CD":
                df = df.T
            df.index.name = "LSOA11CD"
            df = df.reindex(target_index)
            return df.T
        
        self.ev_reg_private_df = reindex_df(self.ev_reg_private_df)
        self.ev_reg_company_df = reindex_df(self.ev_reg_company_df)
        self.ev_reg_total_df = reindex_df(self.ev_reg_total_df)
    
    def _fill_missing_private_ev_data(self):
        # Fill instances where total and company are known but private is unknown
        BOOL = self.ev_reg_total_df.notna() & self.ev_reg_private_df.isna() & self.ev_reg_company_df.notna()
        self.ev_reg_private_df[BOOL] = self.ev_reg_total_df[BOOL] - self.ev_reg_company_df[BOOL]

        # Fill instances where total is known but private and company are unknown
        BOOL = self.ev_reg_total_df.notna() & self.ev_reg_private_df.isna() & self.ev_reg_company_df.isna()
        self.ev_reg_private_df[BOOL] = round(0.5*self.ev_reg_total_df[BOOL])
        self.ev_reg_df = self.ev_reg_private_df
        return self.ev_reg_df
    
    def _interpolate_missing_data(self): # New
        def set_first_nan_to_one(df):
            for col in df.columns:
                col_data = df[col]
                if not col_data.isna().all():
                    # Find first index where value is NaN
                    first_nan_idx = col_data.index[col_data.isna()][0] if col_data.isna().any() else None
                    if first_nan_idx == col_data.index[0]:
                        df.at[first_nan_idx, col] = 1
            return df

        self.ev_reg_df = set_first_nan_to_one(self.ev_reg_df)
        # Interpolate missing values
        self.ev_reg_df = self._interpolate_df(self.ev_reg_df)
        return self.ev_reg_df
    
    def _interpolate_df(self, df: pd.DataFrame) -> pd.DataFrame: # New
        # Identify columns that are not all NaN
        not_all_nan_cols = df.columns[~df.isna().all()]

        # Apply interpolation only to those columns
        interpolated_df = df.copy()
        interpolated_df[not_all_nan_cols] = df[not_all_nan_cols].apply(self._interpolate_col, axis=0)
        return interpolated_df
    
    def _interpolate_col(self, col: pd.Series) -> pd.Series:
        dates = self._calculate_date_range(self._calculate_time_index(col, 'start'), self._calculate_time_index(col, 'end'))
        mask = ~col.isna().values
        if mask.any():
            xp = dates[mask]
            fp = col[mask]
            x = dates
            interpolated = np.round(np.interp(x, xp, fp))
            return pd.Series(data=interpolated, index=col.index)
        else:
            return pd.Series(data=np.nan, index=col.index)
    
    def _calculate_date_range(self, t0: float, t1: float, sample_rate=4) -> np.array:
        # sample_rate = 4 implies quarterly data
        return np.linspace(t0, t1, int((t1-t0)*sample_rate) + 1)
    
    def _calculate_time_index(self, data: pd.Series | pd.DataFrame, position: str = 'start') -> float:
        """Returns a float representing the time index from a quarterly datetime index string."""
        if position == 'start':
            index_str = data.index[0]
        elif position == 'end':
            index_str = data.index[-1]
        else:
            raise ValueError("`position` must be either 'start' or 'end'.")

        year = int(index_str[:4])
        quarter = index_str[-2:]

        quarter_map = {'Q1': 0.0, 'Q2': 0.25, 'Q3': 0.5, 'Q4': 0.75}
        if quarter not in quarter_map:
            raise ValueError(f"Invalid quarter format in index: {quarter}")

        return year + quarter_map[quarter]
    
    def _convert_quarter_index_to_float(self, index: pd.Index) -> float:
        year, quarter = index.split(' Q')
        year = int(year)
        quarter = int(quarter)
        quarter_to_float = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75}
        return year + quarter_to_float[quarter]
    
    def _annualise_data_last(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Year'] = df.index.astype(int)
        return df.groupby('Year').last()
    
    def _filter_by_lad(self, LAD: str) -> dict:
        """Extracts data for a specific Local Authority District (LAD) using the LSOA lookup table."""
        lad_data_dict = {}
        lad_lsoa_ids = self.lsoa_lookup.loc[self.lsoa_lookup['LAD22NM'] == LAD, 'LSOA11CD'].tolist()
        for key, df in self.annual_data_dict.items():
            lad_data_dict[key] = df.loc[:, df.columns.intersection(lad_lsoa_ids)]
        return lad_data_dict
    

