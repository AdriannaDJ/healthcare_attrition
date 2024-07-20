from flask import Flask, render_template, request
import pickle
import pandas as pd

# create app
app = Flask(__name__)

# load assets
with open('training/model.pkl', 'rb') as f: 
    model=pickle.load(f)    
with open('training/scaler.pkl', 'rb') as f: 
    scaler=pickle.load(f)
with open('training/choices.pkl', 'rb') as f: 
    choices=pickle.load(f)
with open('training/ohe.pkl', 'rb') as f: 
    ohe=pickle.load(f)
    
# get list of categorical and numerical columns
cat_cols=[col for col in choices.keys() if choices[col]!=None]
cont_cols=[col for col in choices.keys() if choices[col]==None]
# get all possible categorical features
cat_feature_names=ohe.get_feature_names_out()

@app.route("/", methods=['GET', 'POST'])
def index():
    ''' 
        function to return result only if POST request
    '''
    if request.method == 'POST':
        input_data=request.form.to_dict()
        result=predict(input_data)
        return render_template('index.html', choices=choices, result=result)
    else: 
        return render_template('index.html', choices=choices)

def predict(input_data): 
    '''
        helper function for making prediction
    '''
    # create input DataFrame
    input_df=pd.DataFrame([input_data])
    # flask returns values in forms as strings by default
    X=input_df[cont_cols].astype('float')
    # use OHE on categorical variables
    X_cat=ohe.transform(input_df[cat_cols])
    # add categorical features to input data
    X[cat_feature_names]=X_cat
    # scale input data
    X_transformed=scaler.transform(X)
    # make prediction
    output=model.predict(X_transformed)
    # should only return first and only element
    return output[0]

if __name__ == '__main__':
    app.run()