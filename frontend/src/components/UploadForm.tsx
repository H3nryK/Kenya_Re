import React, { useState } from 'react';
import { useHistory } from 'react-router-dom';

interface UploadFormProps {
  setQuotation: (quotation: any) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({ setQuotation }) => {
  const [proposal, setProposal] = useState<File | null>(null);
  const [financialStatement, setFinancialStatement] = useState<File | null>(null);
  const [license, setLicense] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const history = useHistory();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!proposal || !financialStatement) {
      alert('Please upload both the proposal and financial statement.');
      return;
    }

    setIsLoading(true);

    const formData = new FormData();
    formData.append('proposal', proposal);
    formData.append('financial_statement', financialStatement);
    if (license) {
      formData.append('license', license);
    }

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setQuotation(data.quotation);
      history.push('/success');
    } catch (error) {
      console.error('Error:', error);
      history.push('/error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
      <div className="mb-4">
        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="proposal">
          Proposal Form (PDF)
        </label>
        <input
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
          id="proposal"
          type="file"
          accept=".pdf"
          onChange={(e) => setProposal(e.target.files?.[0] || null)}
          required
        />
      </div>
      <div className="mb-4">
        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="financial-statement">
          Audited Financial Statement (PDF)
        </label>
        <input
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
          id="financial-statement"
          type="file"
          accept=".pdf"
          onChange={(e) => setFinancialStatement(e.target.files?.[0] || null)}
          required
        />
      </div>
      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="license">
          Operations License (PDF, Optional)
        </label>
        <input
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
          id="license"
          type="file"
          accept=".pdf"
          onChange={(e) => setLicense(e.target.files?.[0] || null)}
        />
      </div>
      <div className="flex items-center justify-between">
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
          type="submit"
          disabled={isLoading}
        >
          {isLoading ? 'Uploading...' : 'Submit'}
        </button>
      </div>
    </form>
  );
};

export default UploadForm;