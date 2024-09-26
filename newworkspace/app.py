import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from automation import AdvancedUnderwritingSystem
import tempfile

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the underwriting system
pi_rating_guide_path = os.path.join('documents', 'rating_guide.pdf')
underwriting_system = AdvancedUnderwritingSystem(pi_rating_guide_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'proposal' not in request.files or 'financial_statement' not in request.files:
        return jsonify({'error': 'Missing required files'}), 400

    proposal = request.files['proposal']
    financial_statement = request.files['financial_statement']
    license_file = request.files.get('license')

    if not allowed_file(proposal.filename) or not allowed_file(financial_statement.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    if license_file and not allowed_file(license_file.filename):
        return jsonify({'error': 'Invalid license file type'}), 400

    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as proposal_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as financial_temp:
            proposal.save(proposal_temp.name)
            financial_statement.save(financial_temp.name)

            license_temp = None
            if license_file:
                license_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                license_file.save(license_temp.name)

            # Process the application
            result = underwriting_system.process_application(proposal_temp.name, financial_temp.name, license_temp.name if license_temp else None)

        # Clean up temporary files
        os.unlink(proposal_temp.name)
        os.unlink(financial_temp.name)
        if license_temp:
            os.unlink(license_temp.name)

        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'quotation': result.get('quotation'),
                'message': 'Application processed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('message', 'An error occurred during processing')
            }), 400

    except Exception as e:
        app.logger.error(f"Error processing application: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/download-quotation/<string:quotation_id>', methods=['GET'])
def download_quotation(quotation_id):
    try:
        quotation_path = underwriting_system.get_quotation_pdf(quotation_id)
        if not quotation_path or not os.path.exists(quotation_path):
            return jsonify({'error': 'Quotation not found'}), 404

        return send_file(quotation_path, as_attachment=True, download_name=f"quotation-{quotation_id}.pdf")
    except Exception as e:
        app.logger.error(f"Error downloading quotation: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)