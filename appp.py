
import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))
#with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
#    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
         PROD_CD=flask.request.form['PROD_CD']
         SLSMAN_CD=flask.request.form['SLSMAN_CD']
         PLAN_MONTH=flask.request.form['PLAN_MONTH']
         PLAN_YEAR=flask.request.form['PLAN_YEAR']
         TARGET_IN_EA=flask.request.form['TARGET_IN_EA']
        # Extract the input

        # Make DataFrame for model
        
         input_variables=pd.DataFrame([[PROD_CD,SLSMAN_CD,PLAN_MONTH,PLAN_YEAR,TARGET_IN_EA]],columns=['PROD_CD','SLSMAN_CD','PLAN_MONTH','PLAN_YEAR','TARGET_IN_EA'],dtype=float,index=['input'])
   

        # Get the model's prediction
         prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
         return flask.render_template('main.html',original_input={'PROD_CD':PROD_CD,'SLSMAN_CD':SLSMAN_CD,'MONTH':PLAN_MONTH,'YEAR':PLAN_YEAR,'TARGET':TARGET_IN_EA},result=prediction,)
                                     

if __name__ == '__main__':
    app.run()