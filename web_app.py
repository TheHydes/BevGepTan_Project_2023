import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

st.write("""
# Movie Popularity Prediction
## This app can predict a movie's popularity based on metadata.
The dataset contains movie statistics of 4800 movies.
""")

link = '[Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')


# Get user inputs
def user_input_features():
    Budget = st.sidebar.slider('budget', 90000, 360000, 180000)
    Revenue = st.sidebar.slider('revenue', 800000, 2500000, 1250000)
    Runtime = st.sidebar.slider('runtime', 130, 200, 165)
    Vote_average = st.sidebar.slider('vote_average', 1, 10, 5)
    Vote_count = st.sidebar.slider('vote_count', 150, 1500, 750)
    Release_date = st.sidebar.date_input('release_date', datetime.date(2011,1,1))
    Genre = st.sidebar.multiselect('genre',"Action")
    data = {'budget': Budget,
            'revenue': Revenue,
            'runtime': Runtime,
            'vote_average': Vote_average,
            'vote_count': Vote_count,
            'release_date': Release_date,
            'genre': Genre
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Create Plotly plot
columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count']

# create a new DataFrame with the selected columns
df_movie = df[columns]

# Multiply columns, for a better spectacle
df_movie['runtime'] = df_movie['runtime'] * 10000
df_movie['budget'] = df_movie['budget'] * 5
df_movie['vote_average'] = df_movie['vote_average'] * 100000
df_movie['vote_count'] = df_movie['vote_count'] * 1000

# Convert the first row of the DataFrame to a list
y = df_movie.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Movie Features')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_final_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Movie Popularity')
prediction = int(np.round(prediction, 0))
st.title(prediction)
