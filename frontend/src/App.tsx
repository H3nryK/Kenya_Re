import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import DocumentUploadWithStyles from './components/DocumentUpload';

const App: React.FC = () => {

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <div className="container mx-auto py-8">
          <Routes>
            <Route path="/" element={<DocumentUploadWithStyles />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;