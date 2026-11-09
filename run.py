import torch
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from src.data import load_data, split_and_scale, get_dates
from src.model import TempNN, TempLSTM, eval, predict
from src.train import train_model, make_windows
from src.plot import barplot, lineplot, prediction_plot

def main():
    ds_uci = load_data()
    print("Data Loaded")

    metrics = pd.DataFrame(columns=["model", "r2", "rmse", "skill"])

    features = ["t2m", "rolling_3", "rolling_7"]

    X_train, y_train, X_val, y_val, X_test, y_test = split_and_scale(ds_uci, features, "target_t2m")

    metrics.loc[len(metrics)] = {
        "model": "baseline",
        "r2": r2_score(y_test, X_test["t2m"]),
        "rmse": root_mean_squared_error(y_test, X_test["t2m"]),
        "skill": 0
    }

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    metrics.loc[len(metrics)] = {
        "model": "lin_reg_3",
        "r2": r2_score(y_test, reg.predict(X_test)),
        "rmse": root_mean_squared_error(y_test, reg.predict(X_test)),
        "skill": 1 - (root_mean_squared_error(y_test, reg.predict(X_test))/metrics["rmse"][0])
    }

    EXPERIMENTS = [
        {"name": "nn_3",  "model": TempNN,   "features": ["t2m", "rolling_3", "rolling_7"], "history": 1},
        {"name": "nn_7",  "model": TempNN,   "features": ["t2m", "rolling_3", "rolling_7", "u10", "v10", "msl", "tp"], "history": 1},
        {"name": "lstm",  "model": TempLSTM, "features": ["t2m", "u10", "v10", "msl", "tp"], "history": 7},
    ]

    pred_temps = []

    for idx, exp in enumerate(EXPERIMENTS):
        name, model, features, history = exp.values()

        model = model(input_dim=len(features))

        X_train, y_train, X_val, y_val, X_test, y_test, X_scaler, y_scaler = split_and_scale(ds=ds_uci, FEATURES=features, scale=True)

        X_train, y_train = make_windows(X_train, y_train, history)
        X_val, y_val = make_windows(X_val, y_val, history)
        X_test, y_test = make_windows(X_test, y_test, history)


        model, epoch = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        pred_temps.append(predict(model, X_test, y_scaler))

        test_r2, test_rmse, test_skill = eval(model, X_test, y_test, y_scaler, baseline=metrics["rmse"][0])

        metrics.loc[len(metrics)] = {
        "model": name,
        "r2": test_r2,
        "rmse": test_rmse,
        "skill": test_skill
        }

    # Metrics plots
    barplot(metrics["model"], metrics["r2"], metrics["model"], "Models", "r2", "Compared r2 score Between Models", "r2")
    print("r2 plot saved")
    barplot(metrics["model"], metrics["rmse"], metrics["model"], "Models", "RMSE (C)", "Compared RMSE Between Models", "rmse")
    print("rmse plot saved")
    barplot(metrics["model"], metrics["skill"], metrics["model"], "Models", "Skill", "Compared Skill Between Models", "skill")
    print("skill plot saved")

    # Predicted temp vs real temp
    dates, y_true = get_dates(ds_uci)
    dates, y_true = dates.iloc[history - 1:], y_true[history - 1:]
    prediction_plot(dates, y_true, pred_temps[2], "Date", "Temperature (°C)", "LSTM 7-day forecast vs. observed temperature at UCI (test period)", "lstm_test")
    print("prediction plot saved")

    # Skill against lead time

    lead_metrics = pd.DataFrame(columns=["model", "lead", "rmse", "skill"])
    max_history = max(exp["history"] for exp in EXPERIMENTS)

    for step in range(1, 15):
        ds_lead = ds_uci.copy()
        ds_lead["target_t2m"] = ds_lead["t2m"].shift(valid_time=-step)

        # Baseline and linear regression
        features = ["t2m", "rolling_3", "rolling_7"]

        X_train, y_train, X_val, y_val, X_test, y_test = split_and_scale(ds_lead, features, "target_t2m", gap=step)
        X_test, y_test = X_test.iloc[max_history - 1:], y_test[max_history - 1:]

        baseline_rmse = root_mean_squared_error(y_test, X_test["t2m"])

        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg_rmse = root_mean_squared_error(y_test, reg.predict(X_test))

        lead_metrics.loc[len(lead_metrics)] = {
            "model": "lin_reg_3",
            "lead": step,
            "rmse": reg_rmse,
            "skill": 1 - (reg_rmse / baseline_rmse)
        }

        # Neural networks
        for idx, exp in enumerate(EXPERIMENTS):
            name, model_class, features, history = exp.values()

            X_train, y_train, X_val, y_val, X_test, y_test, X_scaler, y_scaler = split_and_scale(ds=ds_lead, FEATURES=features, scale=True, gap=step)

            X_train, y_train = make_windows(X_train, y_train, history)
            X_val, y_val = make_windows(X_val, y_val, history)
            X_test, y_test = make_windows(X_test, y_test, history)
            X_test, y_test = X_test[max_history - history:], y_test[max_history - history:]

            torch.manual_seed(0)
            model = model_class(input_dim=len(features))

            model, epoch = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            test_r2, test_rmse, test_skill = eval(model, X_test, y_test, y_scaler, baseline=baseline_rmse)

            lead_metrics.loc[len(lead_metrics)] = {
                "model": name,
                "lead": step,
                "rmse": test_rmse,
                "skill": test_skill
            }
        print(f"Skill vs lead time day {step} completed")

    lineplot(lead_metrics["lead"], lead_metrics["skill"], lead_metrics["model"], "Lead time (days)", "Skill vs. persistence", "Skill vs. Lead Time", "skill_lead")
    print("Skill vs lead time day plot saved")

if __name__ == "__main__":
    main()