import glob
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath="./data/"):
    DATA_DIR = filepath

    temp_files = sorted(
        glob.glob(f"{DATA_DIR}**/*.nc", recursive=True)
    )
    ds = xr.open_mfdataset(
        temp_files,
        combine="by_coords"
    )

    lat_uci, lon_uci = 33.64, -117.84
    ds_uci = ds.sel(latitude=lat_uci, longitude=lon_uci, method="nearest")

    ds_uci["rolling_3"] = ds_uci["t2m"].rolling(valid_time=3).mean() 
    ds_uci["rolling_7"] = ds_uci["t2m"].rolling(valid_time=7).mean()
    ds_uci["target_t2m"] = ds_uci["t2m"].shift(valid_time=-3) # target 3 days in the future

    return ds_uci

def split_and_scale(ds, FEATURES, TARGET, seed=222, scale=False):
    """
    Splits and rescales data into a training, validation, and test set

    Args:
        ds (Xarray.Dataset): The dataset being split
        FEATURES (list): The names of the features being used as the predictors
        TARGET (str): Target variable being predicted
        seed (int): Random seed for train_test_split()
        scale (bool): Rescales the data if specified

    Return:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    ds = ds.dropna(dim="valid_time")

    # Test train split
    X = ds.to_dataframe()[FEATURES]
    y = ds[TARGET].values

    X_train, X_split, y_train, y_split = train_test_split(X, y, random_state=seed, shuffle=False, test_size=0.5)

    X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, random_state=seed, shuffle=False, test_size=0.5)

    if scale:
        X_scaler = StandardScaler().fit(X_train)

        X_train_sc = X_scaler.transform(X_train)
        X_val_sc = X_scaler.transform(X_val)
        X_test_sc = X_scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train_clean = y_train.reshape(-1, 1)

        y_scaler.fit(y_train_clean)
        y_train_sc = y_scaler.transform(y_train_clean)
        y_val_sc = y_scaler.transform(y_val.reshape(-1, 1))
        y_test_sc = y_scaler.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test

