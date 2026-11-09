import glob
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath="./data/"):
    """
    Loads the concatenated netCDF data with some feature engineering

    Args:
        filepath (str): directory where the data is held

    Return:
        An Xarray Dataset at UCI's grid cell with two temperature (C) rolling averages (3 and 7 days) and the target temperature
    """
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

    ds_uci["t2m"] = ds_uci["t2m"] - 273.15

    ds_uci["rolling_3"] = ds_uci["t2m"].rolling(valid_time=3).mean() 
    ds_uci["rolling_7"] = ds_uci["t2m"].rolling(valid_time=7).mean()
    ds_uci["target_t2m"] = ds_uci["t2m"].shift(valid_time=-7) # target 7 days in the future

    return ds_uci

def split_and_scale(ds, FEATURES, TARGET="target_t2m", gap=7, seed=222, scale=False):
    """
    Splits and rescales data into a training, validation, and test set

    Args:
        ds (Xarray.Dataset): The dataset being split
        FEATURES (list): The names of the features being used as the predictors
        TARGET (str): Target variable being predicted
        seed (int): Random seed for train_test_split()
        scale (bool): Rescales the data if specified
        gap (int): adds a gap in the validation and test data to ensure no data leakage when predicting on rolling averages

    Return:
        X_train, y_train, X_val, y_val, X_test, y_test, (if scale=true) X_scaler and y_scaler
    """
    ds = ds.dropna(dim="valid_time")

    # Test train split
    X = ds.to_dataframe()[FEATURES]
    y = ds[TARGET].values

    X_train, X_split, y_train, y_split = train_test_split(X, y, random_state=seed, shuffle=False, test_size=0.5)

    X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, random_state=seed, shuffle=False, test_size=0.5)

    X_val, y_val = X_val.iloc[gap:], y_val[gap:]
    X_test, y_test = X_test.iloc[gap:], y_test[gap:]


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

        return X_train_sc, y_train_sc, X_val_sc, y_val_sc, X_test_sc, y_test_sc, X_scaler, y_scaler

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_dates(ds, seed=222, gap=7):
    """
    Gets the dates from the split being tested

    Args:
        ds (Xrray Dataset)
        seed (int)
        gap (int)

    Return:
        2 numpy arrays: one with the dates and one with the true y over the dates
    """
    ds = ds.dropna(dim="valid_time")
    
    # Test train split
    X = ds["valid_time"].to_series()
    y = ds["target_t2m"].values

    _, X_split, _, y_split = train_test_split(X, y, random_state=seed, shuffle=False, test_size=0.5)

    _, dates, _, y_true = train_test_split(X_split, y_split, random_state=seed, shuffle=False, test_size=0.5)

    dates, y_true = dates.iloc[gap:], y_true[gap:]

    return dates, y_true