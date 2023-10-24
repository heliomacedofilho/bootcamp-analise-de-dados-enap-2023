import requests
import pickle
import streamlit as st
st.set_page_config(page_title="Óbitos por Covid-19 🦠 no Brasil 📍", page_icon= '🦠', layout="wide")

def load_data():
    figures = {'Coroplético': {'Região': None, 'Mesorregião': None, 'Microrregião': None, 'Município': None}, 
               'Dispersão': {'Região': None, 'Mesorregião': None, 'Microrregião': None, 'Município': None}}
    de_para = {'choropleth': 'Coroplético', 'regiao': 'Região', 'mesorregiao': 'Mesorregião',
               'microrregiao': 'Microrregião', 'municipio': 'Município', 'scatter_geo': 'Dispersão'}
    pickle_filenames = ['choropleth-regiao.pkl', 'choropleth-mesorregiao.pkl', 'choropleth-microrregiao.pkl',
                        'choropleth-municipio.pkl', 'scatter_geo-regiao.pkl', 'scatter_geo-mesorregiao.pkl', 
                        'scatter_geo-microrregiao.pkl', 'scatter_geo-municipio.pkl']
    url = 'https://github.com/heliomacedofilho/bootcamp-analise-de-dados-enap-2023/raw/main/bootcamp/project/data/app/'
    for filename in pickle_filenames:
        type_of_map, intraregion = filename.rstrip('.pkl').split('-')
        response = requests.get(f'{url}{filename}', stream='True')
        figures[de_para[type_of_map]][de_para[intraregion]] = pickle.load(response.raw)
    return figures

figures = load_data()
st.markdown('# Óbitos por Covid-19 🦠 no Brasil 📍')
st.markdown("---")
type_of_map = st.sidebar.selectbox('Qual o tipo de mapa a representar os dados?',
                                   ('Coroplético', 'Dispersão'))
intraregion = st.selectbox('Qual a malha geográfica do Brasil a ser considerada?',
                          ('Região', 'Mesorregião', 'Microrregião'))
st.plotly_chart(figures[type_of_map][intraregion]);
