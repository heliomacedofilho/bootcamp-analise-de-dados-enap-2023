#Instalação
# !pip install --upgrade scikit-learn --quiet
# !pip install --upgrade nltk --quiet
# !pip install --upgrade openpyxl --quiet
# !pip install mplcairo
# import nltk
# nltk.download('punkt')

# Data Manipulation
import re
import os
import pandas as pd
import numpy as np
# Language Processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import string
# Vectorizers and NLP Models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# Visualization
import matplotlib, mplcairo
print('Default backend: ' + matplotlib.get_backend()) 
matplotlib.use("module://mplcairo.base")
print('Backend is now ' + matplotlib.get_backend())
# IMPORTANT: Import these libraries only AFTER setting the backend
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
# Load Apple Color Emoji font 
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# Coleta de Dados e Preparação
def leitura():
    dfs = []
    folderpath = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    for file in os.listdir(folderpath):
        if file.endswith('.xlsx'):
            dfs.append(pd.read_excel(os.path.join(folderpath, file)))
    
    df = pd.concat(dfs, axis=0).reset_index()
    df = df.rename(columns={'Carimbo de data/hora': 'timestamp', 'Timestamp': 'timestamp', 
                       'Qual emoji representa o seu sentimento quanto ao conteúdo ministrado hoje? [-]': 'sentiment_review',
                       'Como você avalia o seu aprendizado na aula de hoje? [.]': 'learning_review',
                       'Deixe aqui suas sugestões e comentários.': 'full_review'})
    return df[['timestamp', 'sentiment_review', 'learning_review', 'full_review']]

# Limpeza da Tabela
def cleaning_text(sentence):
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercasing 
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## removing numbers
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## removing punctuation
    tokenized_sentence = word_tokenize(sentence) ## tokenizing 
    stop_words = set(stopwords.words('portuguese')) ## defining stopwords
    tokenized_sentence = [w for w in tokenized_sentence 
                                  if not w in stop_words] ## remove stopwords
    #lemmatized_sentence = [WordNetLemmatizer().lemmatize(word, pos = "v")  # v --> verbs for word in tokenized_sentence]
    cleaned_sentence = ' '.join(word for word in tokenized_sentence)
    return cleaned_sentence
def cleaning(df):
    cleaned_df = df.copy()
    cleaned_df['sentiment_review'] = cleaned_df.sentiment_review.map(lambda x: re.sub('[\d\s]+', '', x))
    cleaned_df['learning_review'] = cleaned_df.learning_review.map(lambda x: re.sub('[\d\s]+', '', x))
    cleaned_df['sentiment_review_score'] = cleaned_df.sentiment_review.map(lambda x: {'😖': 1, '🙁': 2, '😐': 3, '😃': 4, '🤩': 5}.get(x))
    cleaned_df['learning_review_score'] = cleaned_df.learning_review.map(lambda x: {'😖': 1, '🙁': 2, '😐': 3, '😃': 4, '🤩': 5}.get(x))
    cleaned_df['length_review'] = cleaned_df.full_review.map(lambda x: len(x) if isinstance(x, str) else 0)
    cleaned_df["full_review"] = cleaned_df.full_review.fillna('')
    cleaned_df["full_review_cleaned"] = cleaned_df["full_review"].apply(cleaning_text)
    return cleaned_df

#Gerar Linha do Tempo (Visualização)
def draw_timeline(cleaned_df, colunm_name):
    review_evolution = cleaned_df.copy()
    review_evolution[colunm_name] = review_evolution[colunm_name].map({'😃': '✔️', '😐': '😐', '❌': '😖', '🙁': '❌', '🤩': '✔️'})
    review_evolution['date'] = review_evolution.timestamp.dt.date
    review_evolution = pd.DataFrame(review_evolution.groupby('date')[colunm_name].value_counts(normalize = True)).unstack()
    review_evolution.columns = review_evolution.columns.droplevel(level=0).map({'😃': '😃', '😐': '😐', '😖': '😖', '🙁': '😟', '🤩': '😍', '✔️': '✔️', '❌': '❌'})
    fig, ax = plt.subplots(figsize=(15,10))
    for emoji in review_evolution.columns:
        sns.lineplot(x=review_evolution.index, y=emoji, data=review_evolution, linestyle='--').set(title=colunm_name)
    for emoji in review_evolution.columns:
        for idx, percent in enumerate(review_evolution.loc[:, emoji]):
            plt.annotate(emoji, xy=(review_evolution.index[idx], percent), 
                         fontproperties = prop, fontsize=20)
    plt.savefig(f'{colunm_name}-timeline.png')

# Análise
def summarize_week(cleaned_df, column):
    return pd.DataFrame(cleaned_df[column].value_counts(normalize = True).map(lambda x: f'{x:.2%}'))
def create_vectorized_reviews(df):
    vectorizer = TfidfVectorizer(max_df = 0.75, max_features = 5000, ngram_range=(2,2))
    vectorized_reviews = pd.DataFrame(vectorizer.fit_transform(df["full_review_cleaned"]).toarray(),
                                     columns = vectorizer.get_feature_names_out())
    return vectorizer, vectorized_reviews
def train_lda(vectorized_reviews, n_components = 3):
    lda = LatentDirichletAllocation(n_components = n_components)
    return lda, lda.fit_transform(vectorized_reviews)
def get_document_mixture(filtered_df, document_mixture, n_topics):
    return pd.DataFrame(pd.DataFrame(document_mixture).map(lambda x: '{:.2%}'.format(x)).values,
                        columns = [f"Topic {i+1}" for i in range(n_topics)],
                        index = filtered_df['full_review_cleaned'])
def topic_word(vectorizer, model, topic, topwords, with_weights = True):
    topwords_indexes = topic.argsort()[:-topwords - 1:-1]
    if with_weights == True:
        topwords = [(vectorizer.get_feature_names_out()[i], round(topic[i],2)) for i in topwords_indexes]
    if with_weights == False:
        topwords = [vectorizer.get_feature_names_out()[i] for i in topwords_indexes]
    return topwords
def get_topic_mixture(vectorizer, model, topwords):
    matrix = []
    for idx, topic in enumerate(model.components_):
        topics_weights = topic_word(vectorizer, model, topic, topwords)
        matrix += [[idx, topic_weight[0], topic_weight[1]] for topic_weight in topics_weights]
    return pd.DataFrame(matrix, columns = ['Topic', 'Bigram', 'Weight'])
def analysis(cleaned_df, comparison, n_topics = 3):
    filtered_df = cleaned_df[comparison(cleaned_df["learning_review_score"])].copy()
    vectorizer, vectorized_reviews = create_vectorized_reviews(filtered_df)
    lda, document_mixture = train_lda(vectorized_reviews, n_topics)
    doc_mixture = get_document_mixture(filtered_df, document_mixture, n_topics)
    top_mixture = get_topic_mixture(vectorizer, lda, topwords = 15)
    return filtered_df, vectorized_reviews, doc_mixture, top_mixture