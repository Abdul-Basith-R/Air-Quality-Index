import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging



def create_dt_index(dataframe):
    dataframe = dataframe.drop(columns='To Date')
    dataframe['From Date'] = pd.to_datetime(dataframe['From Date'])
    dataframe = dataframe.rename(columns={'From Date': 'datetime'})
    return dataframe.set_index('datetime')

def value_dropper(df):
    # dropping values that have more than 80% null values
    null_values = null_value_percentage(df)
    droppable_values = [key for key, value in null_values.items() if value > 0.80 ]
    new_df = df.drop(droppable_values,axis=1)
    return new_df


def null_value_percentage(df):
    null_percentage = {}
    for i in list(df.columns):
        null_percentage[i] = round(len(df[df[i].isnull()])/len(df),3)
    return null_percentage    


def column_merger(df1,column_name1,column_name2,convertion = 1):
    df1[column_name2] = df1[column_name2] * convertion
    df1[column_name1].fillna(df1[column_name2],inplace = True)
    df1.drop([column_name2], axis=1, inplace=True)

def outlier_detection(df1,remove = False):
    # Example for PM2.5 column
    df1 = df1.reset_index().drop(["city"],axis=1).copy()
    outliers_per_column = {}
    unique_outliers = set()
    for value in df1.columns:
        q1 = df1[value].quantile(0.25)
        q3 = df1[value].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_per_column[value] = len(df1[(df1[value] < lower_bound) | (df1[value] > upper_bound)])
        if remove == True:
            for j in np.where((df1[value] < lower_bound) | (df1[value] > upper_bound))[0]:
                if j not in unique_outliers:
                    unique_outliers.add(j)
    if remove == True:
        df1.drop(unique_outliers,inplace= True)
        return df1
    return outliers_per_column,len(unique_outliers)
def log_transform_columns(df, columns):
    df[columns] = np.log1p(df[columns])
    return df
def interpolate_values(df, interpolate_columns):
    for i in range(len(interpolate_columns)):
        df[interpolate_columns[i]].interpolate(method='linear',inplace=True)
    return df
def main(df):

    logger = logging.getLogger(__name__)
    logger.info('Prerpocessing data by removing unwanted data and filling data using interpolation')
    df = create_dt_index(df)


    column_merger(df,"NOx (ppb)","NOx (ppm)",1000)
    column_merger(df,'BP (mmHg)','BP (mg/m3)',0.00750062)
    column_merger(df,'WD (degree)','WD (deg)')
    column_merger(df,'RH (%)','RH ()')
    column_merger(df,'WS (m/s)','VWS (m/s)')

    df = value_dropper(df)


    columns_to_log_transform = df.columns.difference(['city'])
    df = log_transform_columns(df, columns_to_log_transform)


    ds = outlier_detection(df,remove=True)
    ds.set_index(ds['datetime'],drop=True,inplace=True)
    ds.drop("datetime",inplace=True,axis=1)


    interpolate_columns = [col for col, percentage in null_value_percentage(df).items() if percentage < 0.3]
    interpolate_columns.remove('city')
    interpolate_columns.remove('NOx (ppb)')

    ds = interpolate_values(ds, interpolate_columns)

    ds['NOx (ppb)'].interpolate(method='pad', inplace=True)


    train_mice = ds.copy(deep=True)
    columns = train_mice.columns


    mice_imputer = IterativeImputer(random_state = 42, max_iter=10)
    train_mice = mice_imputer.fit_transform(train_mice)
    train_mice = pd.DataFrame(train_mice,columns=columns)
    train_mice = train_mice.set_index(ds.index)
    return train_mice


if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data","processed","data.csv"))
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    data = main(df)
    data.to_csv(os.path.join("data","interim","data.csv"))




