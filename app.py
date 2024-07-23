from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import pickle
import json
import os
import re

app = Flask(__name__)

# Load ML model and preprocessing objects
with open('training/model.pkl', 'rb') as f: 
    model = pickle.load(f)
with open('training/scaler.pkl', 'rb') as f: 
    scaler = pickle.load(f)
with open('training/choices.pkl', 'rb') as f: 
    choices = pickle.load(f)
with open('training/ohe.pkl', 'rb') as f: 
    ohe = pickle.load(f)

# SQLite database connection
conn = sqlite3.connect('employee_data.db')
cursor = conn.cursor()

# Function to create SQLite table and insert data
def create_and_insert():
    # Read CSV into pandas DataFrame
    df = pd.read_csv('Resources/watson_healthcare_modified.csv')

    # Remove unnecessary columns
    df = df.drop(columns=['EmployeeID', 'EmployeeCount', 'StandardHours', 'TrainingTimesLastYear', 'MonthlyRate', 'DailyRate', 'HourlyRate', 'Over18'])
    
    # Rename columns
    df.rename(columns={'Age':'Age (18-60)', 'DistanceFromHome':'Distance From Home (miles)', 'Education':'Education (1-5)', 'EnvironmentSatisfaction':'Env. Satisfaction (1-4)',
                   'JobInvolvement':'Job Involvement (1-4)', 'JobLevel':'Job Level (1-5)', 'JobSatisfaction':'Job Satisfaction (1-4)', 'PerformanceRating':'Performance Rating (1-4)',
                   'RelationshipSatisfaction':'Relationship Satisfaction (1-4)', 'Shift':'Shift (1-3)', 'WorkLifeBalance':'WorkLifeBalance (1-4)'}, inplace=True)
    
    # Create SQL table
    df.to_sql('employee_data', conn, if_exists='replace', index=False)

# Call the function to create table and insert data
create_and_insert()
    
# get list of categorical and numerical columns
cat_cols = [col for col in choices.keys() if choices[col] != None]
cont_cols = [col for col in choices.keys() if choices[col] == None]
# get all possible categorical features
cat_feature_names = ohe.get_feature_names_out()

def camel_case_to_spaces(name):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).title()

@app.route("/", methods=['GET', 'POST'])
def index():
    result = None
    input_data = None
    if request.method == 'POST':
        input_data = request.form.to_dict()
        result = predict(input_data)
        return render_template('index.html', choices=choices, result=result, input_data=json.dumps(input_data), camel_case_to_spaces=camel_case_to_spaces)
    else: 
        return render_template('index.html', choices=choices, camel_case_to_spaces=camel_case_to_spaces)

def predict(input_data): 
    # create input DataFrame
    input_df = pd.DataFrame([input_data])
    # flask returns values in forms as strings by default
    X = input_df[cont_cols].astype('float')
    # use OHE on categorical variables
    X_cat = ohe.transform(input_df[cat_cols])
    # add categorical features to input data
    X[cat_feature_names] = X_cat
    # scale input data
    X_transformed = scaler.transform(X)
    # make prediction
    output = model.predict(X_transformed)
    return output[0]

if __name__ == '__main__':
    app.run(debug=True)



