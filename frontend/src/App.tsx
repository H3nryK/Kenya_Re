import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, redirect } from 'react-router-dom';
import UploadForm from './components/UploadForm';
import ErrorPage from './components/ErrorPage';
import SuccessPage from './components/SuccessPage';

const App: React.FC = () => {
  const [quotation, setQuotation] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <div className="container mx-auto py-8">
          <Routes>
            <Route exact path="/">
              <UploadForm setQuotation={setQuotation} />
            </Route>
            <Route path="/error">
              <ErrorPage />
            </Route>
            <Route path="/success">
              {quotation ? <SuccessPage quotation={quotation} /> : <redirect to="/" />}
            </Route>
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;