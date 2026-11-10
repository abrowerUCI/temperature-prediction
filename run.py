import os
import torch
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from src.data import load_data, split_and_scale, get_dates
from src.model import TempNN, TempLSTM, eval, predict
from src.train import train_model, make_windows
from src.plot import barplot, lineplot, prediction_plot

SEEDS = [0, 1, 2, 3, 4]

MODEL_NAMES = {
    "baseline": "Baseline (persistence)",
    "lin_reg_3": "Linear regression, 3 variables",
    "nn_3": "Neural network, 3 variables",
    "nn_7": "Neural network, 7 variables",
    "lstm": "LSTM neural network",
}

def main():
    ds_uci = load_data()
    print("Data Loaded")

    lineplot(ds_uci["valid_time"], ds_uci["t2m"], None, "Time (days)", "Temperature (C)", "Temperature at UC Irvine (2009-2010)", "uci_temp")

    metrics = pd.DataFrame(columns=["model", "seed", "r2", "rmse", "skill"])

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

    pred_temps = {}

    for idx, exp in enumerate(EXPERIMENTS):
        name, model_class, features, history = exp.values()

        X_train, y_train, X_val, y_val, X_test, y_test, X_scaler, y_scaler = split_and_scale(ds=ds_uci, FEATURES=features, scale=True)

        X_train, y_train = make_windows(X_train, y_train, history)
        X_val, y_val = make_windows(X_val, y_val, history)
        X_test, y_test = make_windows(X_test, y_test, history)

        for seed in SEEDS:
            torch.manual_seed(seed)
            model = model_class(input_dim=len(features))

            model, epoch = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            pred_temps[(name, seed)] = predict(model, X_test, y_scaler)

            test_r2, test_rmse, test_skill = eval(model, X_test, y_test, y_scaler, baseline=metrics["rmse"][0])

            metrics.loc[len(metrics)] = {
            "model": name,
            "seed": seed,
            "r2": test_r2,
            "rmse": test_rmse,
            "skill": test_skill
            }
        print(f"{name} trained on {len(SEEDS)} seeds")

    summary = metrics.groupby("model", sort=False)[["r2", "rmse", "skill"]].agg(["mean", "std"])

    # Results tables
    results_table = pd.DataFrame({"Model": summary.index.map(MODEL_NAMES)})
    for metric, column in [("r2", "Test R²"), ("rmse", "Test RMSE (°C)"), ("skill", "Test skill")]:
        results_table[column] = [
            f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}"
            for mean, std in zip(summary[(metric, "mean")], summary[(metric, "std")])
        ]

    os.makedirs("./results", exist_ok=True)
    metrics.astype({"seed": "Int64"}).to_csv("./results/metrics.csv", index=False)
    results_table.to_csv("./results/results_table.csv", index=False)
    print("metrics and results table saved")

    # Metrics plots
    barplot(metrics["model"], metrics["r2"], metrics["model"], "Models", "R$\mathregular{^2}$", "Compared R$\mathregular{^2}$ score Between Models", "r2")
    print("r2 plot saved")
    barplot(metrics["model"], metrics["rmse"], metrics["model"], "Models", "RMSE (C)", "Compared RMSE Between Models", "rmse")
    print("rmse plot saved")
    barplot(metrics["model"], metrics["skill"], metrics["model"], "Models", "Skill vs. persistence", "Compared Skill Between Models", "skill")
    print("skill plot saved")

    # Predicted temp vs real temp
    dates, y_true = get_dates(ds_uci)
    dates, y_true = dates.iloc[history - 1:], y_true[history - 1:]
    prediction_plot(dates, y_true, pred_temps[("lstm", SEEDS[0])], "Date", "Temperature (°C)", "LSTM 7-day forecast vs. observed temperature at UCI (test period)", "lstm_test")
    print("prediction plot saved")

    # Skill against lead time

    lead_metrics = pd.DataFrame(columns=["model", "lead", "seed", "rmse", "skill"])
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

            for seed in SEEDS:
                torch.manual_seed(seed)
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
                    "seed": seed,
                    "rmse": test_rmse,
                    "skill": test_skill
                }
        print(f"Skill vs lead time day {step} completed")

    lineplot(lead_metrics["lead"], lead_metrics["skill"], lead_metrics["model"], "Lead time (days)", "Skill vs. persistence", "Skill vs. Lead Time", "skill_lead")
    print("Skill vs lead time day plot saved")

if __name__ == "__main__":
    main()