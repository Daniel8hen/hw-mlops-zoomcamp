import os
from datetime import datetime
import pandas as pd
# import batch
import boto3

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def generate_mock_data_based_on_q3():
    data = [
            (None, None, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
            (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)
    categorical = columns[:2]

    # Expected logic
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def save_data():
    print(f"Going to save the file in: {input_file}")
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

df_input = generate_mock_data_based_on_q3()
print(df_input['duration'].sum())
input_file = os.getenv("INPUT_FILE_PATTERN")
input_file = f'{input_file}'.format(year=2021, month=1)
print(input_file)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

# save_data()
# df = pd.read_parquet("s3://nyc-duration/in/2021-01.parquet")
# print(df_input['duration'].sum())
# print(df.groupby().sum('prediction'))


bucket = 'nyc-duration'
#Make sure you provide / in the end
s3 = boto3.client('s3')
all_objects = s3.list_objects(Bucket = bucket)
for obj in all_objects:
    print(obj)

