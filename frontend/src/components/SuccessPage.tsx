import React from 'react';

interface SuccessPageProps {
  quotation: {
    premium: number;
    coverage: number;
    policy_number: string;
  };
}

const SuccessPage: React.FC<SuccessPageProps> = ({ quotation }) => {
  return (
    <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
      <h1 className="text-4xl font-bold mb-4">Quotation Generated Successfully</h1>
      <div className="mb-4">
        <h2 className="text-2xl font-semibold mb-2">Policy Details</h2>
        <p><strong>Policy Number:</strong> {quotation.policy_number}</p>
        <p><strong>Premium:</strong> ${quotation.premium.toFixed(2)}</p>
        <p><strong>Coverage:</strong> ${quotation.coverage.toFixed(2)}</p>
      </div>
      <div className="mb-4">
        <a
          href={`/api/download-quotation/${quotation.policy_number}`}
          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded inline-block"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download Quotation
        </a>
      </div>
    </div>
  );
};

export default SuccessPage;