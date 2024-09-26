import PyPDF2
import pandas as pd
import numpy as np
import re
import spacy
import joblib
import cv2
import os
import pytesseract
import uuid

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from google.cloud import vision

# Building paths inside the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('documents', 'credentials.json')

class AdvancedDocumentProcessor:
    def __init__(self):
        self.vision_client = vision.ImageAnnotatorClient()

    def process_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_image = page.to_image(resolution=300)
                image_bytes = page_image.original.tobytes()
                
                image = vision.Image(content=image_bytes)
                response = self.vision_client.document_text_detection(image=image)
                text += response.full_text_annotation.text + "\n\n"
        return text

class AdvancedDataExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_key_info(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        
        info = {
            'cedant_name': self.extract_entity(doc, "ORG"),
            'broker_name': self.extract_entity(doc, "ORG", exclude=info.get('cedant_name')),
            'limit_of_indemnity': self.extract_money(text, r'Limit of [Ii]ndemnity:?\s*(\$?[\d,]+(?:\.\d{2})?)', 0),
            'occupation': self.extract_field(text, r'Occupation:?\s*(.+)'),
            'estimated_annual_income': self.extract_money(text, r'Estimated annual (?:fee|income):?\s*(\$?[\d,]+(?:\.\d{2})?)', 0),
            'num_qualified_staff': self.extract_int(text, r'Number of qualified staff:?\s*(\d+)'),
            'num_unqualified_staff': self.extract_int(text, r'Number of unqualified staff:?\s*(\d+)'),
            'years_in_business': self.extract_int(text, r'Years in [Bb]usiness:?\s*(\d+)'),
            'industry': self.extract_field(text, r'Industry:?\s*(.+)'),
        }

        return info
    
    def extract_entity(self, doc, label: str, exclude: Optional[str] = None) -> Optional[str]:
        for ent in doc.ents:
            if ent.label_ == label and ent.text != exclude:
                return ent.text
        return None

    def extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_int(self, text: str, pattern: str) -> Optional[int]:
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def extract_money(self, text: str, pattern: str, default: float = 0) -> float:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(re.sub(r'[^\d.]', '', match.group(1)))
        return default

class AdvancedRatingEngine:
    def __init__(self, pi_rating_guide_path: str):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.pi_rating_guide = self.load_pi_rating_guide(pi_rating_guide_path)
    
    def load_pi_rating_guide(self, file_path: str) -> Dict[str, Any]:
        # Placeholder
        return {"base_rate": 0.01, "risk_multipliers": {"Low": 1.0, "Medium": 1.5, "High":2.0}}

    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        risk_type_map = {'Low': 0, 'Medium': 1, 'High': 2}
        industry_map = {'Technology': 0, 'Manufacturing': 1, 'Healthcare': 2, 'Finance': 3, 'Other': 4}
        
        feature_array = np.array([
            features['sum_insured'],
            risk_type_map.get(features['risk_type'], 1),
            industry_map.get(features['industry'], 4),
            features['years_in_business'],
            features['revenue']
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
        score = 0
        
        # Calculate score based on limit of indemnity
        for threshold in self.rating_guide["risk_factors"]["limit_of_indemnity"]:
            if features['limit_of_indemnity'] > threshold["threshold"]:
                score += threshold["factor"]
                break
        
        # Calculate score based on years in business
        for threshold in self.rating_guide["risk_factors"]["years_in_business"]:
            if features['years_in_business'] < threshold["threshold"]:
                score += threshold["factor"]
                break
        
        # Calculate score based on industry
        industry_factor = self.rating_guide["risk_factors"]["industry"].get(
            features['industry'], self.rating_guide["risk_factors"]["industry"]["Other"]
        )
        score += industry_factor
        
        # Determine rating based on score
        if score > 3.5:
            return 'High'
        elif score > 2.5:
            return 'Medium'
        else:
            return 'Low'

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
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("PROFESSIONAL IDEMNITY QUOTATION", styles['Title']))
        elements.append(Spacer(1, 12))

        data = [
            ["REINSURED", "FIRST ASSURANCE"],
            ["BROKER", quotation_data['broker_name']],
            ["INSURED", quotation_data['cedant_name']],
            ["OCCUPATION", quotation_data['occupation']],
            ["LIMIT OF INDEMNITY", f"${quotation_data['limit_of_indemnity']:,.2f}"],
            ["PREMIUM", f"${quotation_data['premium']:,.2f}"],
            ["POLICY NUMBER", quotation_data['policy_number']]
        ]

        table = Table(data, colWidths=[120, 120, 120, 120])

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
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

        table.setStyle(style)

        elements.append(table)

        doc.build(elements)

class AdvancedUnderwritingSystem:
    def __init__(self, pi_rating_guide_path: str):
        self.doc_processor = AdvancedDocumentProcessor()
        self.data_extractor = AdvancedDataExtractor()
        self.rating_engine = AdvancedRatingEngine(pi_rating_guide_path)
        self.quotation_generator = EnhancedQuotationGenerator()
        self.quotations_folder = 'quotations'
        os.makedirs(self.quotations_folder, exist_ok=True)

    def process_application(self, proposal_path: str, audit_path: str):
        try:
            proposal_text = self.doc_processor.process_pdf(proposal_path)
            audit_text = self.doc_processor.process_pdf(audit_path)
            license_text = self.doc_processor.process_pdf(license_path) if license_path else ""

            combined_text = f"{proposal_text}\n{audit_text}\n{license_text}"
            key_info = self.data_extractor.extract_key_info(combined_text)

            rating = self.rating_engine.predict_rating(key_info)
            quotation = self.prepare_quotation(key_info, rating)

            quotation_id = str(uuid.uuid4())
            quotation_path = os.path.join(self.quotations_folder, f"{quotation_id}.pdf")
            self.quotation_generator.generate_pdf(quotation, quotation_path)

            return {
                'status': 'success',
                'quotation': {**quotation, 'id': quotation_id},
                'message': 'Application processed successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f"An error occurred during application processing: {str(e)}"
            }
        
    def prepare_quotation(self, key_info: Dict[str, Any], rating: str) -> Dict[str, Any]:
        try:
            base_rate = self.rating_engine.rating_guide["base_rate"]
            risk_multiplier = 1.5 if rating == 'High' else 1.2 if rating == 'Medium' else 1.0
            industry_factor = self.rating_engine.rating_guide["risk_factors"]["industry"].get(
                key_info['industry'], self.rating_engine.rating_guide["risk_factors"]["industry"]["Other"]
            )
            
            premium = key_info['limit_of_indemnity'] * base_rate * risk_multiplier * industry_factor
            
            return {
                **key_info,
                'premium': premium,
                'policy_number': f"POL-{uuid.uuid4().hex[:8].upper()}",
                'rating': rating
            }

        except Exception as e:
            print(f"An Error occured while preparing the quotation: {str(e)}")
            raise

    def get_quotation_pdf(self, quotation_id: str) -> Optional[str]:
        quotation_path = os.path.join(self.quotations_folder, f"{quotation_id}.pdf")
        return quotation_path if os.path.exists(quotation_path) else None

def generate_dummy_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    sum_insured = np.random.uniform(100000, 2000000, num_samples)
    risk_type = np.random.choice(['Low', 'Medium', 'High'], num_samples)
    industry = np.random.choice(['Technology', 'Manufacturing', 'Healthcare', 'Finance', 'Other'], num_samples)
    years_in_business = np.random.randint(1, 50, num_samples)
    revenue = np.random.uniform(1000000, 100000000, num_samples)

    X = np.column_stack((sum_insured, 
                         np.where(risk_type == 'Low', 0, np.where(risk_type == 'Medium', 1, 2)),
                         np.where(industry == 'Technology', 0, np.where(industry == 'Manufacturing', 1, np.where(industry == 'Healthcare', 2, np.where(industry == 'Finance', 3, 4)))),
                         years_in_business,
                         revenue))

    # Generate target variable (you might want to adjust this logic based on your domain knowledge)
    y = np.where((sum_insured > 1000000) | (revenue > 5000000), 2,  # High risk
                 np.where((sum_insured > 500000) | (years_in_business < 5), 1,  # Medium risk
                          0))  # Low risk

    return X, y

# Usage
pi_rating_guide_path = os.path.join(BASE_DIR / 'documents/rating_guide.pdf')
system = AdvancedUnderwritingSystem(pi_rating_guide_path)

# Generate and train on dummy data
x, y = generate_dummy_data(1000)
system.rating_engine.train(x, y)
system.rating_engine.save_model("dummy_rating_model.joblib")

print("Trained and saved dummy model.")

# Check if a trained model exists and load it
model_path = "dummy_rating_model.joblib"
if os.path.exists(model_path):
    system.rating_engine.load_model(model_path)
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found. Using fallback rating method.")

# processing teh application
system.process_application(
    os.path.join(BASE_DIR / 'newworkspace/documents/proposal.pdf'),
    os.path.join(BASE_DIR / 'newworkspace/documents/audit.pdf')
)

print("Advanced quotation generated successfully.")