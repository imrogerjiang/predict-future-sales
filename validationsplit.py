import pandas as pd
from sklearn.metrics import mean_squared_error


def split_train_val(ds, split_month=None):
    '''
    Split dataset into 
        train (date_block_num < month)
        test (date_block_num >= month)
    '''
    if split_month is None:
        split_month = ds["date_block_num"].max()
    
    train_ds = ds[ds["date_block_num"] < split_month]
    val_ds = ds[ds["date_block_num"] == split_month]
    
    return train_ds, val_ds

def split_x_y(ds, y_col):
    return ds.drop(columns=y_col), ds[y_col]

def val_score(predictions, val_ds):
    '''
    Calculates the root mean squared error between prediction and validation
    '''
    if type(predictions)==type(val_ds)==pd.core.frame.DataFrame:
        temp = val_ds.rename(columns={"item_cnt_month":"actual_sales"}).merge(
            predictions.rename(columns={"item_cnt_month":"predicted_sales"}), 
            on=["shop_id", "item_id"], 
            how="outer")
        temp["predicted_sales"] = temp["predicted_sales"].fillna(0).clip(0, 20)
        temp["actual_sales"] = temp["actual_sales"].fillna(0).clip(0, 20)
        return metrics.mean_squared_error(temp["actual_sales"], temp["predicted_sales"])**0.5
    elif type(predictions)==type(val_ds)==pd.core.series.Series:
        temp_pred = predictions.fillna(0).clip(0, 20)
        temp_val = val_ds.fillna(0).clip(0, 20)
        return mean_squared_error(temp_pred, temp_val)**0.5