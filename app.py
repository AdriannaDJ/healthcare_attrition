from flask import Flask, render_template, request
import pickle
import pandas as pd
import json
from wordcloud import WordCloud
import os
import re

# create app
app = Flask(__name__)

# load assets
with open('training/model.pkl', 'rb') as f: 
    model = pickle.load(f)    
with open('training/scaler.pkl', 'rb') as f: 
    scaler = pickle.load(f)
with open('training/choices.pkl', 'rb') as f: 
    choices = pickle.load(f)
with open('training/ohe.pkl', 'rb') as f: 
    ohe = pickle.load(f)
    
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
        generate_wordcloud(result)
    return render_template('index.html', choices=choices, result=result, input_data=json.dumps(input_data), camel_case_to_spaces=camel_case_to_spaces)

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

def generate_wordcloud(prediction):
    text = prediction
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Save the word cloud image
    wordcloud_path = os.path.join('static', 'wordcloud.png')
    wordcloud.to_file(wordcloud_path)

if __name__ == '__main__':
    app.run()


