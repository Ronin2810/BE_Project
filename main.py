import streamlit as st
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tenseal as ts
import numpy as np
import pickle

custom_stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40
vectorizer = Word2Vec.load('vectorizer.model')
W_xh = np.load('W_xh.npy')
W_hh = np.load('W_hh.npy')
W_hy = np.load('W_hy.npy')
b_h  = np.load('b_h.npy')
b_y  = np.load('b_y.npy')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in custom_stopwords]
    return tokens

def vectorize_text(tokens):
    vectors = [vectorizer.wv[word] for word in tokens if word in vectorizer.wv]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return [0] * 1000

def apply_ckks_vector(entry):
    st.session_state['ckks_vec_state'] = ts.ckks_vector(context, entry)
    return st.session_state['ckks_vec_state']

def ckks_dot(vector,W_xh):
  result=[]
  for row in W_xh:
    result.append(vector.dot(row))
  return np.array(result)

def decrypt_dot(lst):
  n_lst = []
  for row in lst:
    n_lst.append(row.decrypt()[0])
  return np.array(n_lst)

def softmax(x):
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)

def forward_enc(inputs,W_xh,W_hh,W_hy,b_h,b_y):
        h_t = np.zeros((128, 1)) # 128 us the hidden size
        hidden_states = [] # not needed
        outputs = []
        for x in inputs:
            tp = decrypt_dot(ckks_dot(x,W_xh))
            tp = tp.reshape(-1,1)
            h_t = np.tanh(tp + np.dot(W_hh, h_t) + b_h)
            hidden_states.append(h_t) # not needed
            y_t = np.dot(W_hy, h_t) + b_y
            outputs.append(softmax(y_t))
        return outputs

def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

# Function to send email
def send_email(sender_email, sender_password, receiver_email, subject, message):
    # Set up the SMTP server
    smtp_server = "smtp.gmail.com"  # For Gmail
    smtp_port = 587  # For Gmail

    # Create a MIMEText object to represent the email message
    email_message = MIMEMultipart()
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message['Subject'] = subject
    email_message.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Start TLS encryption
    server.login(sender_email, sender_password)

    # Send the email
    server.sendmail(sender_email, receiver_email, email_message.as_string())

    # Close the connection
    server.quit()

# Function to generate random alphanumeric code
def generate_code_encryption():
    if 'encryption_key_state' not in st.session_state:
        st.session_state['encryption_key_state'] = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    else :
        return st.session_state['encryption_key_state']

def generate_code_decryption():
    if 'decryption_key_state' not in st.session_state:
        st.session_state['decryption_key_state'] = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    else :
        return st.session_state['decryption_key_state']




# Streamlit UI components
st.title("BE Project")

encryption_key=generate_code_encryption()
transcript=''
with st.form("encryption_form"):
    # Step 1: User Inputs Text
    transcript = st.text_input("Enter your transcript:")

    # Step 2: User Enters Email Address
    email = st.text_input("Enter your email address:")

    # Step 3: Send Encryption Key Button
    send_key_button = st.form_submit_button("Send Encryption Key")

# Send encryption key via email

if send_key_button:
    encryption_key=generate_code_encryption()
    send_email('adityapatildev2810@gmail.com', 'iopw ecxc jrgi fubf', email, "Encryption Key", f"Your Encryption key is: {encryption_key}")
    st.success("Encryption Key sent to your Email!")

encryption_key_input=''
with st.form("encryption_key_form"):
    # Step 4: User Enters Encryption Key
    encryption_key_input = st.text_input("Enter your key:")
    
    # Step 5: Encrypt Button
    encrypt_button = st.form_submit_button("Encrypt and Predict Diagnosis")

decryption_key=generate_code_decryption()

# Perform encryption
ckks_vec = np.array([])
if encrypt_button:
    if encryption_key_input==encryption_key and encryption_key!='':
        # Vectorize and encrypt the transcript here
        tokens = preprocess_text(transcript)
        vector = vectorize_text(tokens)
        ckks_vec = apply_ckks_vector(vector)
        st.success("Encryption Successful!")

        send_email('adityapatildev2810@gmail.com', 'iopw ecxc jrgi fubf', email, "Decryption Key", f"Your Decryption key is: {decryption_key}")
        st.success("Decryption Key for inference sent to your Email!")
    else:
        st.error("Wrong Encryption Key")

decryption_key_input = ''
with st.form("prediction_form"):
    # Step 6: User Enters Decryption Key
    decryption_key_input = st.text_input("Enter Decryption key:")

    # Step 7: Decrypt Button
    decrypt_button = st.form_submit_button("Decrypt Diagnosis")


# Perform decryption and display prediction
if decrypt_button:

    if decryption_key_input==decryption_key and decryption_key!="":
    # Assuming decryption process and prediction retrieval here

        outputs = forward_enc([st.session_state['ckks_vec_state']],W_xh,W_hh,W_hy,b_h,b_y)
        outs = outputs[-1].reshape(40)
        # predicted_class = [np.argmax(outs)]
        # enc_ans = apply_ckks_vector(predicted_class)

        predicted_classes = np.array(np.argsort(outputs[-1])[-5:][::-1]) 
        enc_ans = apply_ckks_vector(predicted_classes)
        
        with open('class_map.pkl', 'rb') as pickle_file:
            class_map = pickle.load(pickle_file)
        
        # value_to_find = int(enc_ans.decrypt()[0])
        # key_found = get_key_from_value(class_map, value_to_find)
        # prediction = key_found

        predictions = []
        for i in range(5):
            value_to_find = int(enc_ans.decrypt()[i])
            key_found = get_key_from_value(class_map, value_to_find)
            predictions.append(key_found)


        st.success("Inference Completed!")
    # Colab Code here
        st.success("Decryption successful!")
        # st.write("Diagnosis:", prediction)
        st.write("Diagnosis (Top 5):", predictions)
    else:
        st.error("Wrong Decryption Key")
