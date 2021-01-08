from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
run = Run.get_context()
client = ExplanationClient.from_run(run)

ds = TabularDatasetFactory.from_delimited_files("https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv", validate=True, include_path=False, infer_column_types=True, set_column_types=None, separator=',', header=True, partition_format=None, support_multi_line=False, empty_as_string=False)

def transform(dataframe):
    
    le = LabelEncoder()  # Label Enconder From Sklearn
    
    # Select all categorcial features
    categorical_features = list(dataframe.columns[dataframe.dtypes == object])    
    
    # Apply Label Encoding on all categorical features
    return dataframe[categorical_features].apply(lambda x: le.fit_transform(x))

def clean_data(data):

    df = data.copy()
    df[['agent', 'company']] = df[['agent', 'company']].fillna(0.0)
    df['country'].fillna(data.country.mode().to_string(), inplace=True)
    df['children'].fillna(round(data.children.mean()), inplace=True)
    df = df.drop(df[(df.adults+df.babies+df.children)==0].index)
    df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')

    df_cleaned = df.copy()
    df_cleaned['room'] = 0
    df_cleaned.loc[df_cleaned['reserved_room_type'] == df_cleaned['assigned_room_type'] , 'room'] = 1
    df_cleaned['net_cancelled'] = 0
    df_cleaned.loc[df_cleaned['previous_cancellations'] > df_cleaned['previous_bookings_not_canceled'] , 'net_cancelled'] = 1
    df_cleaned = df_cleaned.drop(['reservation_status','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                                'arrival_date_month','assigned_room_type','reserved_room_type','reservation_status_date',
                                'previous_cancellations','previous_bookings_not_canceled'],axis=1)
    
    df_cleaned_t = df_cleaned.copy()
    df_cleaned_t = transform(df_cleaned_t)
    df_cleaned[['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type']] = df_cleaned_t[['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type']]

    x = df_cleaned.drop(['is_canceled'], axis=1) 
    y = df_cleaned['is_canceled']
    
    return x, y

x, y = clean_data(ds)

feature_names = list(x.columns)
feature_names = np.append(feature_names, ["is_canceled"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/modelht.pkl')
    
    #model_name
    model_file_name = 'modelht.pkl'
    
    # register the model
    run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
    original_model = run.register_model(model_name='model_explain',model_path='original_model.pkl')

    # Explain predictions on your local machine
    tabular_explainer = TabularExplainer(model, x_train, features=feature_names)
    global_explanation = tabular_explainer.explain_global(x_test)

    # The explanation can then be downloaded on any compute
    comment = 'Global explanation on regression model trained on bank marketing campaing dataset'
    client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)    

if __name__ == '__main__':
    main()
