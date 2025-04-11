# Foundations of Data Science 
# CS - 712 
# Project Report – Team Delta 
# Title: Detection and Classification of Fake Job Recruitment Posting 
# Jannatul Ferdousi Rajoni (200530077)
# Varun Biren Majithia (200532798)
# Samarth Subhash Deshpande (200532782)

# Date : 11/04/2025


import streamlit as st
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import contractions
import nltk
from lime.lime_text import LimeTextExplainer
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ First Streamlit command
st.set_page_config(page_title="Job Fraud Detector", page_icon="🔍")

# ✅ Initialize session state keys
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'title': '',
        'company_profile': '',
        'description': '',
        'requirements': '',
        'benefits': '',
        'employment_type': '',
        'required_experience': '',
        'required_education': '',
        'function': '',
        'industry': '',
        'telecommuting': 0,
        'has_company_logo': 1,
        'has_questions': 0
    }

# ✅ Text Preprocessor Class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.english_vocab = set(nltk.corpus.words.words())
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
    def is_gibberish(self, word):
        if re.fullmatch(r"[a-zA-Z]\d+[a-zA-Z]", word) or re.fullmatch(r"(.)\1{3,}", word):
            return True
        return word.lower() not in self.english_vocab and len(word) > 3
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return [self._preprocess_text(text) for text in X]
        
    def _preprocess_text(self, text):
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = contractions.fix(text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return " ".join([self.lemmatizer.lemmatize(word)
                         for word in text.split()
                         if word not in self.stop_words and not self.is_gibberish(word)])

# ✅ Load model
@st.cache_resource
def load_model():
    return joblib.load('fraud_detection_pipeline.pkl')

# ✅ LIME Helper
def predict_for_lime(text_samples):
    base_data = {
        'title': st.session_state.inputs['title'],
        'company_profile': st.session_state.inputs['company_profile'],
        'requirements': st.session_state.inputs['requirements'],
        'benefits': st.session_state.inputs['benefits'],
        'employment_type': st.session_state.inputs['employment_type'],
        'required_experience': st.session_state.inputs['required_experience'],
        'required_education': st.session_state.inputs['required_education'],
        'function': st.session_state.inputs['function'],
        'industry': st.session_state.inputs['industry'],
        'telecommuting': st.session_state.inputs['telecommuting'],
        'has_company_logo': st.session_state.inputs['has_company_logo'],
        'has_questions': st.session_state.inputs['has_questions']
    }
    
    input_list = []
    for text in text_samples:
        modified = base_data.copy()
        modified['description'] = text
        input_list.append(modified)
        
    return model.predict_proba(pd.DataFrame(input_list))  # 2D array output

# ✅ Streamlit UI
st.title("🔍 Job Post Fraud Detection")

st.markdown("""
🚨 **Job Fraud Detection System** 🚨  
Welcome to the Job Fraud Detection System! This tool helps you analyze and detect fraudulent job postings using machine learning. Fill in the details of a job post below, and we'll let you know if it is likely a fraud or not!

### How It Works:
1. Enter job details in the form below 📝  
2. Submit the form to get predictions ⚡  
3. Receive results and insights with LIME explanations 🔍
""")

# ✅ Input Form
with st.form("job_form"):
    st.header("Job Details")
    
    st.session_state.inputs['title'] = st.text_input("Title✨*", value=st.session_state.inputs['title'])
    st.session_state.inputs['company_profile'] = st.text_area("Company Profile🏢*", value=st.session_state.inputs['company_profile'])
    st.session_state.inputs['description'] = st.text_area("Description📝*", value=st.session_state.inputs['description'])
    st.session_state.inputs['requirements'] = st.text_area("Requirements💼", value=st.session_state.inputs['requirements'])
    st.session_state.inputs['benefits'] = st.text_area("Benefits🎉", value=st.session_state.inputs['benefits'])

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.inputs['employment_type'] = st.text_input("Employment Type 📅", value=st.session_state.inputs['employment_type'])
        st.session_state.inputs['required_experience'] = st.text_input("Required Experience🧳", value=st.session_state.inputs['required_experience'])
        st.session_state.inputs['required_education'] = st.text_input("Required Education🎓", value=st.session_state.inputs['required_education'])

    with col2:
        st.session_state.inputs['function'] = st.text_input("Job Function👨‍💻", value=st.session_state.inputs['function'])
        st.session_state.inputs['industry'] = st.text_input("Industry🏭", value=st.session_state.inputs['industry'])
        st.session_state.inputs['telecommuting'] = st.selectbox("Telecommuting🏡", [0, 1], index=st.session_state.inputs['telecommuting'])
        st.session_state.inputs['has_company_logo'] = st.selectbox("Company Logo🖼️", [0, 1], index=st.session_state.inputs['has_company_logo'])
        st.session_state.inputs['has_questions'] = st.selectbox("Has Questions❓", [0, 1], index=st.session_state.inputs['has_questions'])

    submitted = st.form_submit_button("Analyze Post🚀")

# ✅ On Submit
if submitted:
    model = load_model()
    try:
        input_data = pd.DataFrame([st.session_state.inputs])

        # ✅ Make Prediction
        proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # ✅ Display Results
        st.subheader("Results 📊")
        result_text = f"Fraud Probability: {proba:.1%}"
        
        if prediction == 1:
            st.error(f"🚨 **Fraud Detected!** {result_text}")
        else:
            st.success(f"✅ **Legit Post** {result_text}")
        
        st.progress(proba)
        
        # ✅ LIME Explanation
        if st.session_state.inputs['description']:
            st.subheader("Explanation 🔍")
            explainer = LimeTextExplainer(class_names=['Legit', 'Fraud'])
            
            explanation = explainer.explain_instance(
                st.session_state.inputs['description'],
                predict_for_lime,
                num_features=6
            )
            
            # 🔍 List Features
            st.write("### Important Features Contributing to the Prediction:")
            st.write(explanation.as_list())
            
            # 📊 Bar Plot with Plotly
            explanation_dict = explanation.as_list()
            features = [item[0] for item in explanation_dict]
            weights = [item[1] for item in explanation_dict]

            fig = px.bar(
                x=weights,
                y=features,
                orientation='h',
                labels={'x': 'Feature Weight', 'y': 'Features'},
                title='Feature Contribution to Prediction',
                color=weights,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")
