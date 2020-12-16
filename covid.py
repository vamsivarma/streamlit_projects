import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Renaming columns for the sake of readability
new_column_names_dict = {
    'data': 'date',
    'stato': 'state',
    'codice_regione': 'region code',
    'denominazione_regione': 'region name',
    'lat': 'latitude',
    'long': 'longitude',
    'ricoverati_con_sintomi': 'hospitalized_with_symptoms',
    'terapia_intensiva': 'intensive care',
    'totale_ospedalizzati': 'total hospitalized',
    'isolamento_domiciliare': 'home isolation',
    'totale_positivi': 'total positives', 
    'variazione_totale_positivi': 'total positive change',
    'nuovi_positivi': 'new positives',
    'dimessi_guariti': 'discharged healed',
    'deceduti': 'deceased',
    'totale_casi': 'total cases',
    'tamponi': 'tampons',
    'casi_testati': 'total tests',
    'note_it': 'notes in italian',
    'note_en': 'notes in english',
    'codice_provincia': '',
    'denominazione_provincia': 'province name',
    'sigla_provincia': 'province abbreviation'
    
}

# Function for renaming columns and reformmating the date
def preprocess_df(df):
    df.rename(columns = new_column_names_dict, inplace = True)
    df['date'] = pd.to_datetime(df.date).apply(lambda x: x.date())
    df['date'] = pd.to_datetime(df.date)
    
    return df

@st.cache
def get_data(data_path):
    data = pd.read_csv(data_path)
    data_df = preprocess_df(data)

    return data_df

regions_path = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
provinces_path = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"
total_data_path = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"

regions_df = get_data(regions_path)
province_df = get_data(provinces_path)
total_df = get_data(total_data_path)

st.title("Analysis of Covid Dataset")
st.header("Dataset viewer")

st.markdown("Overview of regions data")
st.dataframe(regions_df.head())

st.markdown("Overview of provinces data")
st.dataframe(province_df.head())

st.markdown("Overview of total data")
st.dataframe(total_df.head())

st.markdown("Map for Regions dataset")
st.map(regions_df[["latitude", "longitude"]].dropna(how="any"))

st.markdown("Map for Provinces dataset")
st.map(province_df[["latitude", "longitude"]].dropna(how="any"))





