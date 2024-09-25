import PyPDF2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re
import spacy
import joblib
from typing import Dict, Any

class AdvancedDocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def process_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def process_excel(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path)

class AdvancedDataExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_key_info(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        info = {
            'client_name': None,
            'sum_insured': None,
            'risk_type': None,
            'industry': None,
            'years_in_business': None,
            'claims_history': None
        }

        for ent in doc.ents:
            if ent.label_ == "PERSON" and not info['client_name']:
                info['client_name'] = ent.text
            elif ent.label_ == "MONEY" and not info['sum_insured']:
                info['sum_insured'] = float(re.sub(r'[^\d.]', '', ent.text))
            # Add more entity extraction logic here

        # Use regex for specific patterns
        info['risk_type'] = re.search(r'Risk Type:\s*(.+)', text)
        info['industry'] = re.search(r'Industry:\s*(.+)', text)
        info['years_in_business'] = re.search(r'Years in Business:\s*(\d+)', text)
        info['claims_history'] = re.search(r'Claims History:\s*(.+)', text)

        # Convert matches to strings or None
        for key, value in info.items():
            if isinstance(value, re.Match):
                info[key] = value.group(1) if value else None

        return info

class AdvancedRatingEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        # Convert categorical variables to numerical
        risk_type_map = {'Low': 0, 'Medium': 1, 'High': 2}
        industry_map = {'Technology': 0, 'Manufacturing': 1, 'Healthcare': 2, 'Finance': 3, 'Other': 4}
        
        feature_array = np.array([
            features['sum_insured'],
            risk_type_map.get(features['risk_type'], -1),
            industry_map.get(features['industry'], 4),
            features['years_in_business'],
            len(features['claims_history'].split(',')) if features['claims_history'] else 0
        ]).reshape(1, -1)
        
        return self.scaler.transform(feature_array)

    def train(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        print(f"Model accuracy: {self.model.score(X_test_scaled, y_test):.2f}")

    def predict_rating(self, features: Dict[str, Any]) -> str:
        preprocessed_features = self.preprocess_features(features)
        prediction = self.model.predict(preprocessed_features)
        return ['Low', 'Medium', 'High'][int(prediction[0])]

    def save_model(self, path: str):
        joblib.dump((self.model, self.scaler), path)

    def load_model(self, path: str):
        self.model, self.scaler = joblib.load(path)

class AdvancedQuotationGenerator:
    def generate_pdf(self, quotation_data: Dict[str, Any], output_path: str):
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Reinsurance Quotation")
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, f"Client: {quotation_data['client_name']}")
        c.drawString(100, 700, f"Sum Insured: ${quotation_data['sum_insured']:,.2f}")
        c.drawString(100, 680, f"Risk Type: {quotation_data['risk_type']}")
        c.drawString(100, 660, f"Industry: {quotation_data['industry']}")
        c.drawString(100, 640, f"Years in Business: {quotation_data['years_in_business']}")
        c.drawString(100, 620, f"Claims History: {quotation_data['claims_history']}")
        c.drawString(100, 600, f"Risk Rating: {quotation_data['risk_rating']}")
        c.drawString(100, 580, f"Premium: ${quotation_data['premium']:,.2f}")
        c.save()

class AdvancedUnderwritingSystem:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.doc_processor = AdvancedDocumentProcessor()
        self.data_extractor = AdvancedDataExtractor(self.nlp)
        self.rating_engine = AdvancedRatingEngine()
        self.quotation_generator = AdvancedQuotationGenerator()

    def process_application(self, file_path: str):
        # Process document
        if file_path.endswith('.pdf'):
            text = self.doc_processor.process_pdf(file_path)
        elif file_path.endswith('.xlsx'):
            df = self.doc_processor.process_excel(file_path)
            text = df.to_string()
        else:
            raise ValueError("Unsupported file format")

        # Extract key information
        key_info = self.data_extractor.extract_key_info(text)

        # Predict rating
        rating = self.rating_engine.predict_rating(key_info)

        # Generate quotation
        quotation_data = self.prepare_quotation(key_info, rating)
        self.quotation_generator.generate_pdf(quotation_data, "advanced_quotation.pdf")

    def prepare_quotation(self, key_info: Dict[str, Any], rating: str) -> Dict[str, Any]:
        # More sophisticated premium calculation
        base_rate = 0.01  # 1% base rate
        risk_multiplier = {'Low': 1.0, 'Medium': 1.5, 'High': 2.0}
        industry_factor = {
            'Technology': 1.2,
            'Manufacturing': 1.3,
            'Healthcare': 1.4,
            'Finance': 1.5,
            'Other': 1.1
        }

        premium = (
            key_info['sum_insured'] *
            base_rate *
            risk_multiplier[rating] *
            industry_factor.get(key_info['industry'], 1.1) *
            (1 + 0.01 * int(key_info['years_in_business']))  # 1% discount per year in business
        )

        return {
            **key_info,
            'risk_rating': rating,
            'premium': premium
        }

# Usage
system = AdvancedUnderwritingSystem()

# Train the model (in a real scenario, you'd do this separately with a large dataset)
# X = ... # Your feature matrix
# y = ... # Your target variable
# system.rating_engine.train(X, y)
# system.rating_engine.save_model("rating_model.joblib")

# For subsequent runs, you can load the trained model
# system.rating_engine.load_model("rating_model.joblib")

system.process_application("Fekan Howell - Proposal Form signed and dated 11072024.pdf")
print("Advanced quotation generated successfully.")