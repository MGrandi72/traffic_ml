import pandas as pd                  # Pandas
import numpy as np                   # Numpy
from matplotlib import pyplot as plt # Matplotlib

# Package to implement ML Algorithms
import sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest
# Import MAPIE to calculate prediction intervals
from mapie.regression import MapieRegressor
# To calculate coverage score
from mapie.metrics import regression_coverage_score
# Package for data partitioning
from sklearn.model_selection import train_test_split
# Package to record time
import time
# Module to save and load Python objects to and from files
import pickle 
from datetime import datetime
# Ignore Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

T_pickle = open('bst_traffic.pickle', 'rb') 
T_model = pickle.load(T_pickle)
T_pickle.close()



df = pd.read_csv('Traffic_Volume.csv')
df = df.replace({np.nan: None})

df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday
df['hour'] = df['date_time'].dt.hour


df.loc[df['weekday'] == 0, 'weekday'] = 'Monday'
df.loc[df['weekday'] == 1, 'weekday'] = 'Tuesday'
df.loc[df['weekday'] == 2, 'weekday'] = 'Wednesday'
df.loc[df['weekday'] == 3, 'weekday'] = 'Thursday'
df.loc[df['weekday'] == 4, 'weekday'] = 'Friday'
df.loc[df['weekday'] == 5, 'weekday'] = 'Saturday'
df.loc[df['weekday'] == 6, 'weekday'] = 'Sunday'

df.loc[df['month'] == 1, 'month'] = 'January'
df.loc[df['month'] == 2, 'month'] = 'February'
df.loc[df['month'] == 3, 'month'] = 'March'
df.loc[df['month'] == 4, 'month'] = 'April'
df.loc[df['month'] == 5, 'month'] = 'May'
df.loc[df['month'] == 6, 'month'] = 'June'
df.loc[df['month'] == 7, 'month'] = 'July'
df.loc[df['month'] == 8, 'month'] = 'August'
df.loc[df['month'] == 9, 'month'] = 'September'
df.loc[df['month'] == 10, 'month'] = 'October'
df.loc[df['month'] == 11, 'month'] = 'November'
df.loc[df['month'] == 12, 'month'] = 'December'

df = df.drop(columns = ['date_time'])

df['hour'] = df['hour'].astype(str)

with st.sidebar:
    st.image('traffic_sidebar.jpg', width = 500, 
    caption = "Traffic Volume Predictor")
    st.subheader("Input Features")
    st.write("You can either upload your data file or manually input features.")
    
    with st.expander("Option 1: Upload CSV File"):
        st.header("Option 1: Upload a CSV File")
        input = st.file_uploader("Browse Files")
    
        sample = df.head(5)
        st.write(sample)
        st.warning('Ensure your uploaded file has the same column names and data types as shown above', icon="⚠️")
    
    with st.expander("Option 2: Fill out Form"):
        with st.form("Enter the diamond details manually using the form below"):
            holiday = st.selectbox('Choose whether today is a designated holiday or not', options=df['holiday'].unique())
            temp = st.number_input('Average temperature in Kelvin', min_value=df['temp'].min(), max_value=df['temp'].max(), step=0.5)
            rain_1h = st.number_input('Amount in mm of rain that occured in the hour', min_value=df['rain_1h'].min(), max_value=df['rain_1h'].max(), step=0.1)    
            snow_1h = st.number_input('Amount in mm of snow that occured in the hour', min_value=df['snow_1h'].min(), max_value=df['snow_1h'].max(), step=0.1)
            clouds_all = st.number_input('Percentage of cloud cover', min_value=0, max_value=100, step=1)
            weather_main = st.selectbox('Choose the current weather', options=df['weather_main'].unique())
            month = st.selectbox('Choose month', options=('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
            weekday = st.selectbox('Choose day of the week', options=('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
            hour = st.selectbox('Choose hour', options=('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'))
            submit_button = st.form_submit_button("Submit Form Data")

#USED CHATGPT FOR THE GRADIENT COLORING OF THE TITLE(SYNTAX ASSISTANCE)
gradient_title = """
    <style>
    .gradient-title {
        font-size: 50px;
        font-weight: bold;
        background: linear-gradient(90deg, #ff7f00, #ffeb3b, #4caf50); /* Orange, Yellow, Green */
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
    }
    </style>
    <div class="gradient-title">
        Traffic Volume Predictor
    </div>
"""

# Display the gradient title in Streamlit using st.markdown()
st.markdown(gradient_title, unsafe_allow_html=True)
st.subheader("Utilize our advanced Machine Learning application to predict traffic volume")
st.image('traffic_image.gif', width = 650)

if input is not None:
    st.success('CSV File Uploaded Successfully')
elif submit_button == True:
    st.success('Form Data Submitted Successfully')
else:
    st.warning('Please choose a data input method to proceed')
    

encode_df = df.copy()
encode_df = encode_df.drop(columns=['traffic_volume'])
encode_df2 = encode_df
# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df, columns=['holiday', 'weather_main', 'month', 'weekday', 'hour'])
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)
alpha = st.slider('Select alpha value for prediction interval', min_value=0.01, max_value=0.5, value=0.15, step=0.01) # For 90% confidence level
prediction, intervals = T_model.predict(user_encoded_df, alpha = alpha)
pred_value = prediction[0]
lower_limit = intervals[:, 0]
upper_limit = intervals[:, 1]
 #Ensure limits are within [0, 1]
lower_limit = max(0, lower_limit[0][0])
upper_limit = upper_limit[0][0]

if input is not None:
    inputdf = pd.read_csv(input)
    inputdf = inputdf.replace({np.nan: None})
    inputdf['hour'] = inputdf['hour'].astype(str)
    merge = pd.concat([inputdf,encode_df2], join = 'outer')
    df = pd.get_dummies(merge, columns=['holiday', 'weather_main', 'month', 'weekday', 'hour'])
    df['traffic_volume'] = T_model.predict(df) 
    user_df = df.head(5)
    user_df = user_df.reset_index(drop=True)
    
    usercopy = user_df.copy()
    usercopy = usercopy.drop(columns=['traffic_volume'])
    
    prediction2, intervals2 = T_model.predict(usercopy, alpha = alpha)
    lower_limits = [item[0][0] for item in intervals2]
    upper_limits = [item[1][0] for item in intervals2]
    
    non_negative = [max(value, 0) for value in lower_limits]
    upper_limits = [max(value, 0) for value in upper_limits]
    
    beta = 1-alpha
    
    inputdf['Predicted Volume'] = user_df['traffic_volume']
    inputdf['Lower Limit'] = non_negative
    inputdf['Upper Limit'] = upper_limits
    st.subheader(f"**Prediction Results with**: {(beta)*100:.0f}%"" Confidence Interval")
    st.write(inputdf)
else:
    st.write("## Predicting Prices...")  
    CI = 1-alpha

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{max(0,round(pred_value,0)):.2f}")
    st.write(f"**Confidence Interval**: {(CI)*100:.0f}% [{lower_limit:.2f}, {upper_limit:.2f}]")




st.subheader("Model Performance and Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])

# Tab 1: Visualizing Confusion Matrix
with tab1:
    st.write("### Feature Importance")
    st.image('T_XGB_FI.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 2: Classification Report
with tab2:
    st.write("### Histogram of Residuals")
    st.image('T_XGB_DR.svg')
    st.caption("Histogram of the residuals between actual and predicted.")

# Tab 3: Feature Importance
with tab3:
    st.write("### Predicted vs. Actual")
    st.image('T_XGB_PA.svg')
    st.caption("Plot of predicted vs. actual points.")
    
with tab4:
    st.write("### Coverage Plot")
    st.image('T_XGB_PI.svg')
    st.caption("Plot of confidence intervals given different values.")