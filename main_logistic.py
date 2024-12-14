# base
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import emoji
import scipy

# nltk
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# tf
import tensorflow as tf
# from keras import layers, regularizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    print("main処理開始")

    df_train_origin = pd.read_csv("data/train.csv")
    df_test_origin  = pd.read_csv("data/test.csv")

    print(df_train_origin.describe())
    print( df_train_origin.head(3))
    # 欠損値チェック
    df_train_origin.isna().sum()

    # ターゲット（Rating）の値
    df_train_origin.Rating.unique()

 

   

if __name__ == "__main__":
    main()
