import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate ,TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from pandas.tseries.holiday import USFederalHolidayCalendar


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    #print(df)
    y = df['cnt']
    X = df
    train_indices = X['yr'] == 0
    X_train, y_train = X[train_indices] , y[train_indices]
    X_test, y_test = X[~train_indices], y[~train_indices]
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_dteday_format,
            _fix_missing_dates,
            _fix_new_dates,
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _fix_dteday_format(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

def _fix_missing_dates(df):  
    df.index = df['dteday'] + df['hr'].apply(pd.Timedelta,unit='hour')
    rng = pd.date_range(df.index[0], df.index[-1], freq='1H')
    df = df.reindex(rng)
    return df

def _fix_new_dates(df):
    """
        Year, month, hour, weekday: Come from the index
        holiday: Comes from a Pandas module called "USFederalHolidayCalendar"
        working day: True if is not a holiday and it is between monday to friday
        season, weathersit, temp, atemp, hum, windspeed: take the same record from an hour ago
        casual, registered: Mean of the samples that correspond to the same hour and week day
        cnt: sum of casual and registered
    """
    ### Get holidays in the specific period of time
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2011-01-01', end='2012-12-31').to_pydatetime()

    nan_index = df[df.dteday.isna()].index

    for index in nan_index:
        df.loc[index,['yr', 'mnth', 'hr']] = (0 if int(index.year)==2011 else 1, index.month, index.hour)
        df.loc[index, 'weekday'] = index.dayofweek
        df.loc[index, 'workingday'] = int(index.dayofweek in [0,1,2,3,4] and index not in holidays)
        df.loc[index, 'holiday'] = int(index in holidays)

        sub_index = index
        
        while np.isnan(df.loc[sub_index, 'instant']):
            sub_index = sub_index - pd.Timedelta(1,unit='hr')
            if  not np.isnan(df.loc[sub_index, 'instant']):
                df.loc[index, ['weathersit', 'season', 'temp','atemp', 'hum', 'windspeed']] = df.loc[sub_index, ['weathersit', 'season','temp','atemp', 'hum', 'windspeed']]
                

        df.loc[index, 'casual'] = round(df[(df['hr']==df.loc[index, 'hr']) & (df['weekday']==df.loc[index, 'weekday'])]['casual'].mean())
        df.loc[index, 'registered'] = round(df[(df['hr']==df.loc[index, 'hr']) & (df['weekday']==df.loc[index, 'weekday'])]['registered'].mean())
        df.loc[index, 'cnt'] = df.loc[index, 'casual'] + df.loc[index, 'registered']
    
    return df
    


















# def _fix_pool_quality(df):
#     num_total_nulls = df["PoolQC"].isna().sum()
#     num_nulls_when_poolarea_is_zero = df[df["PoolArea"] == 0]["PoolQC"].isna().sum()
#     assert num_nulls_when_poolarea_is_zero == num_total_nulls
#     num_nulls_when_poolarea_is_not_zero = df[df["PoolArea"] != 0]["PoolQC"].isna().sum()
#     assert num_nulls_when_poolarea_is_not_zero == 0
#     df["PoolQC"] = df["PoolQC"].fillna("NP")
#     return df


# def _fix_misc_feature(df):
#     num_total_nulls = df["MiscFeature"].isna().sum()
#     num_nulls_when_miscval_is_zero = df[df["MiscVal"] == 0]["MiscFeature"].isna().sum()
#     num_nulls_when_miscval_is_not_zero = (
#         df[df["MiscVal"] != 0]["MiscFeature"].isna().sum()
#     )
#     assert num_nulls_when_miscval_is_zero == num_total_nulls
#     assert num_nulls_when_miscval_is_not_zero == 0
#     df["MiscFeature"] = df["MiscFeature"].fillna("No MF")
#     return df


# def _fix_fireplace_quality(df):
#     num_total_nulls = df["FireplaceQu"].isna().sum()
#     num_nulls_when_fireplaces_is_zero = (
#         df[df["Fireplaces"] == 0]["FireplaceQu"].isna().sum()
#     )
#     num_nulls_when_fireplaces_is_not_zero = (
#         df[df["Fireplaces"] != 0]["FireplaceQu"].isna().sum()
#     )
#     assert num_nulls_when_fireplaces_is_zero == num_total_nulls
#     assert num_nulls_when_fireplaces_is_not_zero == 0
#     df["FireplaceQu"] = df["FireplaceQu"].fillna("No FP")
#     return df


# def _fix_garage_variables(df):
#     num_area_zeros = (df["GarageArea"] == 0).sum()
#     num_cars_zeros = (df["GarageCars"] == 0).sum()
#     num_both_zeros = ((df["GarageArea"] == 0) & (df["GarageCars"] == 0.0)).sum()
#     assert num_both_zeros == num_area_zeros == num_cars_zeros
#     for colname in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
#         num_total_nulls = df[colname].isna().sum()
#         num_nulls_when_area_and_cars_capacity_is_zero = (
#             df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)][colname]
#             .isna()
#             .sum()
#         )
#         num_nulls_when_area_and_cars_capacity_is_not_zero = (
#             df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)][colname]
#             .isna()
#             .sum()
#         )
#         assert num_total_nulls == num_nulls_when_area_and_cars_capacity_is_zero
#         assert num_nulls_when_area_and_cars_capacity_is_not_zero == 0
#         df[colname] = df[colname].fillna("No Ga")

#     num_total_nulls = df["GarageYrBlt"].isna().sum()
#     num_nulls_when_area_and_cars_is_zero = (
#         df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)]["GarageYrBlt"]
#         .isna()
#         .sum()
#     )
#     num_nulls_when_area_and_cars_is_not_zero = (
#         df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)]["GarageYrBlt"]
#         .isna()
#         .sum()
#     )
#     assert num_nulls_when_area_and_cars_is_zero == num_total_nulls
#     assert num_nulls_when_area_and_cars_is_not_zero == 0
#     df["GarageYrBlt"].where(
#         ~df["GarageYrBlt"].isna(), other=df["YrSold"] + 1, inplace=True
#     )

#     return df


# def _fix_lot_frontage(df):
#     assert (df["LotFrontage"] == 0).sum() == 0
#     df["LotFrontage"].fillna(0, inplace=True)
#     return df


# def _fix_alley(df):
#     df["Alley"].fillna("NA", inplace=True)
#     return df


# def _fix_fence(df):
#     df["Fence"].fillna("NF", inplace=True)
#     return df


# def _fix_masvnr_variables(df):
#     df = df.dropna(subset=["MasVnrType", "MasVnrArea"])
#     df = df[~((df["MasVnrType"] == "None") & (df["MasVnrArea"] != 0.0))]
#     return df


# def _fix_electrical(df):
#     df.dropna(subset=["Electrical"], inplace=True)
#     return df


# def _fix_basement_variables(df):
#     colnames = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
#     cond = ~(
#         df["BsmtQual"].isna()
#         & df["BsmtCond"].isna()
#         & df["BsmtExposure"].isna()
#         & df["BsmtFinType1"].isna()
#         & df["BsmtFinType2"].isna()
#     )
#     for c in colnames:
#         df[c].where(cond, other="NB", inplace=True)
#     return df


# def _fix_unhandled_nulls(df):
#     df.dropna(inplace=True)
#     return df
