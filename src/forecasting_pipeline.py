"""Final forecasting pipeline for Task 4."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DT="DateTime"; TARGET="Consumption"; HORIZON=168
LAGS=[1,2,3,24,48,168]; ROLL_WINDOWS=[24,168]

def add_features(df):
    data=df[[DT,TARGET]].copy().sort_values(DT).reset_index(drop=True)
    data[DT]=pd.to_datetime(data[DT])
    for lag in LAGS: data[f"lag_{lag}"]=data[TARGET].shift(lag)
    for window in ROLL_WINDOWS:
        data[f"rolling_mean_{window}"]=data[TARGET].shift(1).rolling(window).mean()
        data[f"rolling_std_{window}"]=data[TARGET].shift(1).rolling(window).std()
    data["hour"]=data[DT].dt.hour; data["dayofweek"]=data[DT].dt.dayofweek; data["month"]=data[DT].dt.month
    data["is_weekend"]=(data["dayofweek"]>=5).astype(int)
    data["sin_hour"]=np.sin(2*np.pi*data["hour"]/24); data["cos_hour"]=np.cos(2*np.pi*data["hour"]/24)
    return data

def make_supervised(train_df):
    sup=add_features(train_df).dropna().reset_index(drop=True)
    feature_cols=[c for c in sup.columns if c not in [DT,TARGET]]
    return sup, feature_cols

def make_feature_row(hist_vals, t):
    t=pd.Timestamp(t); d={}
    for lag in LAGS: d[f"lag_{lag}"]=hist_vals[-lag]
    for w in ROLL_WINDOWS:
        arr=np.array(hist_vals[-w:], dtype=float)
        d[f"rolling_mean_{w}"]=arr.mean(); d[f"rolling_std_{w}"]=arr.std(ddof=1)
    d["hour"]=t.hour; d["dayofweek"]=t.dayofweek; d["month"]=t.month; d["is_weekend"]=int(t.dayofweek>=5)
    d["sin_hour"]=np.sin(2*np.pi*t.hour/24); d["cos_hour"]=np.cos(2*np.pi*t.hour/24)
    return d

def recursive_forecast(model, history_df, future_df, feature_cols, horizon=HORIZON):
    hist=list(history_df.sort_values(DT)[TARGET].astype(float).values)
    dates=list(pd.to_datetime(future_df.sort_values(DT)[DT].head(horizon)))
    preds=[]
    for t in dates:
        X=pd.DataFrame([make_feature_row(hist,t)])[feature_cols]
        pred=float(model.predict(X)[0]); preds.append(pred); hist.append(pred)
    return np.array(preds)

def evaluate(y_true,y_pred,scale_mae):
    mae=mean_absolute_error(y_true,y_pred); rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    smape=np.mean(2*np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)))*100
    return {"MAE":mae,"RMSE":rmse,"sMAPE_%":smape,"MASE":mae/scale_mae,"n":len(y_true)}

def run_pipeline(root="."):
    root=Path(root); reports=root/"reports"; reports.mkdir(exist_ok=True, parents=True)
    train=pd.read_csv(root/"data"/"processed"/"electricity_train.csv"); test=pd.read_csv(root/"data"/"processed"/"electricity_test.csv")
    train[DT]=pd.to_datetime(train[DT]); test[DT]=pd.to_datetime(test[DT])
    sup, feature_cols=make_supervised(train)
    model=GradientBoostingRegressor(n_estimators=50, learning_rate=0.06, max_depth=3, random_state=42)
    t0=time.perf_counter(); model.fit(sup[feature_cols].tail(10000), sup[TARGET].tail(10000)); fit_time=time.perf_counter()-t0
    t1=time.perf_counter(); pred=recursive_forecast(model, train, test, feature_cols); pred_time=time.perf_counter()-t1
    y_true=test.sort_values(DT)[TARGET].head(HORIZON).to_numpy(float)
    scale_mae=np.mean(np.abs(train[TARGET].diff(24).dropna()))
    result=evaluate(y_true,pred,scale_mae); result.update({"fit_time_sec":fit_time,"forecast_time_sec":pred_time})
    pd.DataFrame([{**{"model":"Pipeline_GradientBoosting_lags"}, **result}]).to_csv(reports/"pipeline_run_metrics.csv", index=False)
    return result

if __name__ == "__main__": print(run_pipeline())
