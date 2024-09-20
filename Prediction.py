#Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain.llms import Ollama
import os
import datetime
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from apscheduler.schedulers.blocking import BlockingScheduler
import pickle

with open('json', 'r') as g:
    config = json.load(g)

artifacts = config['artifacts']
input_folder = config['input_folder']
output_folder_master = config['output_folder_master']
output_folder_actionable = config['output_folder_actionable']
retrain_folder = config['retrain_folder']

with open(artifacts, 'rb') as f:
    data = pickle.load(f)

model = data['model']
vectorizer = data['vectorizer']
train_df = data['train_df']
X_train_tfidf = data['X_train_tfidf']


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
    

def prediction_using_similarity_and_model(input_error, model, vectorizer, train_df, X_train_tfidf, threshold=0.9):
    preprocessed_input = preprocess(input_error)
    error_vector = vectorizer.transform([preprocessed_input])

    cosine_similarities = cosine_similarity(error_vector, X_train_tfidf).flatten()

    max_similarity_index = cosine_similarities.argmax()
    max_similarity_score = cosine_similarities[max_similarity_index]

    retrain_info = []
    
    if max_similarity_score > threshold:
        return train_df.iloc[max_similarity_index]['Label'], retrain_info
    else:
        prediction = model.predict(error_vector)[0]
        retrain_info.append({'Error': input_error, 'Prediction': prediction})
        return prediction, retrain_info
    
# Find the solutions for the actionable errors
solutions_df = pd.read_csv('C:/Users/narasimha.chandi/OneDrive - HCL TECHNOLOGIES LIMITED/Desktop/Testing/SolutionsFolder/RAGFile.csv')
solutions_df['Error'] = solutions_df['Error'].apply(preprocess)
error_matrix = vectorizer.transform(solutions_df['Error'])

llm = Ollama(model="llama3")

def get_generic_solution(error_message):
    response = llm.invoke(error_message)
    return response

def find_best_solution(error_message, solutions_df, vectorizer, threshold=0.95):
    preprocessed_error_message = preprocess(error_message)
    error_vector = vectorizer.transform([preprocessed_error_message])

    cosine_sim = cosine_similarity(error_vector, error_matrix)
    
    best_match_idx = cosine_sim.argmax()
    best_similarity = cosine_sim[0, best_match_idx]
    
    if best_similarity >= threshold:
        return solutions_df.iloc[best_match_idx]['Solutions']
    else:
        return get_generic_solution(error_message)
    
# Classify and save the results
def classify_and_save(input_file):
    test_df = pd.read_csv(input_file)
    test_df['error_message'] = test_df['message'].apply(extract_system_message)
    test_df['preprocessed_error_message'] = test_df['error_message'].apply(preprocess)
    X_test_tfidf = vectorizer.transform(test_df['preprocessed_error_message']) 
    predictions = []
    retrain_data = []
    for i in test_df['error_message']:
        prediction, retrain_info = prediction_using_similarity_and_model(i, model, vectorizer, train_df, X_train_tfidf)
        predictions.append(prediction)
        if retrain_info:  
            retrain_data.extend(retrain_info)
    prediction_labels = ['Actionable' if i == 1 else 'Non-Actionable' for i in predictions]   
   
    test_df['predictions'] = prediction_labels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if retrain_data:
        retrain_df = pd.DataFrame(retrain_data)
        retrain_df['Prediction'] = retrain_df['Prediction'].apply(lambda x: 'Actionable' if x == 1 else 'Non-Actionable')
        retrain_df['Label'] = 'NaN'
        retrain_df.to_csv(os.path.join(retrain_folder, f"{os.path.basename(input_file).split('.')[0]}_{timestamp}_retrain.csv"), index=False)
    
    actionable_df = test_df[test_df['predictions']=='Actionable']
    if not actionable_df.empty:
        remarks = []
        for i in actionable_df['error_message']:
            solution = find_best_solution(i, solutions_df, vectorizer)
            remarks.append(solution)
        actionable_df['remarks'] = remarks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actionable_df.to_csv(os.path.join(output_folder_actionable, f"{os.path.basename(input_file).split('.')[0]}_{timestamp}_actionable.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder_master, f"{os.path.basename(input_file).split('.')[0]}_{timestamp}_master.csv"), index=False)

# Scheduler to poll the input folder and process files
def poll_input_folder():
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            file_path = file_path.replace("\\", "/")
            print(f"Processing file: {file_path}")
            classify_and_save(file_path)  

scheduler = BlockingScheduler()

scheduler.add_job(poll_input_folder, 'cron', minute='*/5')  

try:
    print("Scheduler started...")
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("Scheduler stopped.")