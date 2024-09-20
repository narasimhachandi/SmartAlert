#Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.utils import shuffle
import json
import os
from sklearn.tree import DecisionTreeClassifier
import pickle


#Preprocessing function
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):  
        return ''  
    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Extract the 'systemMessage' from the JSON string
def extract_system_message(json_str):
    try:
        message_dict = json.loads(json_str)
        
        if 'systemMessage' in message_dict:
            return message_dict['systemMessage']
        
        elif 'errors' in message_dict and isinstance(message_dict['errors'], list):
            return message_dict['errors'][0].get('systemMessage', '')
        
        return 
    except (json.JSONDecodeError, TypeError):
        return ''  
    
with open('json', 'r') as f:
    config = json.load(f)

train_file = config['train_file']
additional_data_folder = config['additional_data_folder']
artifacts = config['artifacts']


# Train or retrain the model
def train_or_retrain(train_file, retrain=False, additional_data_folder=None):
    train_df = pd.read_csv(train_file, sep='\t')
    train_df_original = train_df.copy()
    train_df['Error'] = train_df['Error'].apply(preprocess)
    train_df['Label'] = train_df['Label'].apply(lambda x: 1 if x == 'Actionable' else 0)
    X_train = train_df['Error']
    Y_train = train_df['Label']

    vectorizer = TfidfVectorizer()

    if retrain and additional_data_folder:
        dataframes = []
        for filename in os.listdir(additional_data_folder):
            if filename.endswith('.csv'): 
                file_path = os.path.join(additional_data_folder, filename)
                file_path = file_path.replace("\\", "/")
                df = pd.read_csv(file_path)  
                dataframes.append(df)  
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.drop_duplicates(keep='first')
        train_df_original = pd.concat([train_df_original, combined_df], ignore_index=True)
        combined_df['Error'] = combined_df['Error'].apply(preprocess)
        combined_df['Label'] = combined_df['Label'].apply(lambda x: 1 if x == 'Actionable' else 0)
        X_train_additional = combined_df['Error']
        y_train_additional = combined_df['Label']

        X_train = pd.concat([X_train, X_train_additional], ignore_index=True)
        Y_train = pd.concat([Y_train, y_train_additional], ignore_index=True)
        train_df = pd.concat([train_df, combined_df], ignore_index=True)

        train_df_original.to_csv(train_file, sep='\t', index=False)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = DecisionTreeClassifier()
    model.fit(X_train_tfidf, Y_train)

    return model, vectorizer, train_df, X_train_tfidf


model, vectorizer, train_df, X_train_tfidf = train_or_retrain(train_file=train_file)

with open(artifacts, 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer, 'train_df': train_df, 'X_train_tfidf': X_train_tfidf}, f)

