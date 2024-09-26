import PyPDF2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import inch
import re
import spacy
import joblib
from typing import Dict, Any
import os

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
        
        # Fallback to regex if NER doesn't find the information
        if not info['client_name']:
            client_match = re.search(r'Client Name:\s*(.+)', text)
            info['client_name'] = client_match.group(1) if client_match else "Unknown"
        
        if not info['sum_insured']:
            sum_insured_match = re.search(r'Sum Insured:\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)', text)
            info['sum_insured'] = float(sum_insured_match.group(1).replace(',', '')) if sum_insured_match else 0

        info['risk_type'] = re.search(r'Risk Type:\s*(.+)', text)
        info['industry'] = re.search(r'Industry:\s*(.+)', text)
        info['years_in_business'] = re.search(r'Years in Business:\s*(\d+)', text)
        info['claims_history'] = re.search(r'Claims History:\s*(.+)', text)

        # Convert matches to strings or None
        for key, value in info.items():
            if isinstance(value, re.Match):
                info[key] = value.group(1) if value else None

        # Set default values if information is missing
        info['risk_type'] = info['risk_type'] or "Medium"
        info['industry'] = info['industry'] or "Other"
        info['years_in_business'] = int(info['years_in_business'] or 0)
        info['claims_history'] = info['claims_history'] or "None"

        return info

class AdvancedRatingEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        risk_type_map = {'Low': 0, 'Medium': 1, 'High': 2}
        industry_map = {'Technology': 0, 'Manufacturing': 1, 'Healthcare': 2, 'Finance': 3, 'Other': 4}
        
        feature_array = np.array([
            features['sum_insured'],
            risk_type_map.get(features['risk_type'], 1),
            industry_map.get(features['industry'], 4),
            features['years_in_business'],
            len(features['claims_history'].split(',')) if features['claims_history'] != "None" else 0
        ]).reshape(1, -1)
        
        return feature_array if not self.is_fitted else self.scaler.transform(feature_array)

    def train(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        print(f"Model accuracy: {self.model.score(X_test_scaled, y_test):.2f}")

    def predict_rating(self, features: Dict[str, Any]) -> str:
        if not self.is_fitted:
            return self.fallback_rating(features)
        
        preprocessed_features = self.preprocess_features(features)
        prediction = self.model.predict(preprocessed_features)
        return ['Low', 'Medium', 'High'][int(prediction[0])]

    def fallback_rating(self, features: Dict[str, Any]) -> str:
        sum_insured = features['sum_insured']
        years_in_business = features['years_in_business']
        claims_history = features['claims_history']

        if sum_insured > 1000000 or claims_history != "None":
            return 'High'
        elif sum_insured > 500000 or years_in_business < 5:
            return 'Medium'
        else:
            return 'Low'

    def save_model(self, path: str):
        joblib.dump((self.model, self.scaler, self.is_fitted), path)

    def load_model(self, path: str):
        self.model, self.scaler, self.is_fitted = joblib.load(path)

class EnhancedQuotationGenerator:
    def generate_pdf(self, quotation_data: Dict[str, Any], output_path: str):
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        styles = getSampleStyleSheet()

        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "Reinsurance Quotation")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 70, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

        # Client Information
        client_info = [
            ["Client", quotation_data['client_name']],
            ["Industry", quotation_data['industry']],
            ["Years in Business", str(quotation_data['years_in_business'])],
            ["Claims History", quotation_data['claims_history']]
        ]
        self.create_table(c, client_info, 50, height - 150, 250)

        # Policy Details
        policy_details = [
            ["Sum Insured", f"${quotation_data['sum_insured']:,.2f}"],
            ["Risk Type", quotation_data['risk_type']],
            ["Risk Rating", quotation_data['risk_rating']],
            ["Premium", f"${quotation_data['premium']:,.2f}"]
        ]
        self.create_table(c, policy_details, 300, height - 150, 250)

        # Terms and Conditions
        terms = """
        Terms and Conditions:
        1. This quotation is valid for 30 days from the date of issue.
        2. The premium is subject to change based on any additional information provided.
        3. Coverage is subject to the full terms, conditions, and exclusions of the policy.
        4. This quotation is based on the information provided and may be adjusted if any details change.
        """
        p = Paragraph(terms, styles["BodyText"])
        p.wrapOn(c, width - 100, height)
        p.drawOn(c, 50, height - 350)

        c.save()

    def create_table(self, canvas, data, x, y, width):
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        table = Table(data, colWidths=[width * 0.4, width * 0.6])
        table.setStyle(style)
        table.wrapOn(canvas, width, 400)
        table.drawOn(canvas, x, y)

class AdvancedUnderwritingSystem:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.doc_processor = AdvancedDocumentProcessor()
        self.data_extractor = AdvancedDataExtractor(self.nlp)
        self.rating_engine = AdvancedRatingEngine()
        self.quotation_generator = EnhancedQuotationGenerator()

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

# Check if a trained model exists and load it
model_path = "rating_model.joblib"
if os.path.exists(model_path):
    system.rating_engine.load_model(model_path)
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found. Using fallback rating method.")

# Process the application
system.process_application("Fekan Howell - Proposal Form signed and dated 11072024.pdf")
print("Advanced quotation generated successfully.")