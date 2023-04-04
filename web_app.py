import streamlit as st
import pickle
import numpy as np
import pandas as pd
from plotly import graph_objects as go 
#import plotly.graph_objects as go

st.write("""
# Movie Popularity Prediction
## This app can predict a movie's popularity.
The dataset contains movie statistics of 4800 movies.
""")

link = '[Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')


# Get user inputs
def user_input_features():
    Budget = st.sidebar.slider('budget', 37, 206, 100)
    Release_date = st.sidebar.slider('release_date', 3, 100, 50)
    Revenue = st.sidebar.slider('revenue', 10, 96, 50)
    Runtime = st.sidebar.slider('runtime', -27, 0, -13)
    Vote_average = st.sidebar.slider('vote_average', 2, 99, 50)
    Vote_count = st.sidebar.slider('vote_count', 3, 99, 50)
    data = {'budget': Budget,
            'release_date': Release_date,
            'revenue': Revenue,
            'runtime': Runtime,
            'vote_average': Vote_average,
            'vote_count': Vote_count,
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Create Plotly plot
columns = ['budget', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
df_movie = df.filter(items=columns)
y = df_movie.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Movie Features')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_final_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Movie Popularity')
prediction = int(np.round(prediction, 0))
st.title(prediction)
