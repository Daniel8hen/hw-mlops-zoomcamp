import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
from dateutil.relativedelta import relativedelta
import urllib.request
import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task()
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
        
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task()
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task()
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    """This function will generate dates for train, validation dataset.
    Specifically, it will:
        1. Take in a parameter date: datetime (default: None)
            a. If date is None => will use current date, and will use data from 2 months back as the training data and the data from previous month as validation data.
            b. Else, will get 2 months before the date as the training data, and the previous month as validation data. 
        2. Example: if the date passed is "2021-03-15", 
        training_data = "fhv_tripdata_2021-01.parquet", 
        validation_data = "fhv_trip_data_2021-02.parquet"
    """

    logger = get_run_logger()
    base_url_path = "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"
    data_local_base_path = "./data/fhv_tripdata_"
    data_folder = "./data"
    postfix = ".parquet"
    date_format = "%Y-%m-%d"
    
    if date is None:
        rel_date = datetime.now()
    else:
        rel_date = datetime.strptime(date, date_format)
    
    training_datetime = rel_date + relativedelta(months=-2)
    validation_datetime = rel_date + relativedelta(months=-1)

    train_year, train_month = training_datetime.year, training_datetime.month
    val_year, val_month = validation_datetime.year, validation_datetime.month

    # Since data pattern is month in MM, we want to make sure we don't get "1" as month, for example. Instead, helper function will return "01".
    train_month = get_real_month_value(train_month)
    val_month = get_real_month_value(val_month)

    train_path = f'{data_local_base_path}{train_year}-{train_month}{postfix}'
    val_path = f'{data_local_base_path}{val_year}-{val_month}{postfix}'

    train_url = f'{base_url_path}{train_year}-{train_month}{postfix}'
    valid_url = f'{base_url_path}{val_year}-{val_month}{postfix}'
    logger.info(f"Going to Download data from: {train_url}, {valid_url}")
    train_filename = urllib.request.urlretrieve(train_url, train_path)
    valid_filename = urllib.request.urlretrieve(valid_url, val_path)

    return train_path, val_path

def get_real_month_value(month, max_val=10):
    if month < max_val:
        return f'0{month}'
    else:
        return f'{month}'

@flow()
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save model and DV
    # model
    with open(f'model-{date}.pkl', 'wb') as file:
        pickle.dump(lr, file)
    # dv
    with open(f'dv-{date}.pkl', 'wb') as file:
        pickle.dump(dv, file)

    # run model
    run_model(df_val_processed, categorical, dv, lr)
    

main(date="2021-08-15")
