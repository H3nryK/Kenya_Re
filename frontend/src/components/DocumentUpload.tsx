import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const DocumentUpload = () => {
  const [files, setFiles] = useState({
    proposal: null,
    financial_statement: null,
    operations_license: null
  });
  const [errors, setErrors] = useState({});
  const [isSuccess, setIsSuccess] = useState(false);
  const [quotation, setQuotation] = useState('');

  const onDrop = useCallback((acceptedFiles, fileRejections, event) => {
    const fileType = event.target.name;
    if (acceptedFiles.length > 0) {
      setFiles(prevFiles => ({ ...prevFiles, [fileType]: acceptedFiles[0] }));
      setErrors(prevErrors => ({ ...prevErrors, [fileType]: null }));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleSubmit = (e) => {
    e.preventDefault();
    const newErrors = {};
    if (!files.proposal) newErrors.proposal = 'Proposal is required';
    if (!files.financial_statement) newErrors.financial_statement = 'Financial statement is required';
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
    } else {
      // Simulate API call
      setTimeout(() => {
        setIsSuccess(true);
        setQuotation('Your quotation: $10,000');
      }, 1500);
    }
  };

  if (isSuccess) {
    return (
      <div style={{maxWidth: '400px', margin: '2rem auto', padding: '1.5rem', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'}}>
        <div style={{textAlign: 'center', color: 'green', fontSize: '3rem', marginBottom: '1rem'}}>✓</div>
        <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', textAlign: 'center', marginBottom: '1rem'}}>Success!</h2>
        <p style={{textAlign: 'center', marginBottom: '1rem'}}>Your documents have been uploaded successfully.</p>
        <p style={{textAlign: 'center', fontWeight: 'semibold'}}>{quotation}</p>
      </div>
    );
  }

  return (
    <div style={{maxWidth: '400px', margin: '2rem auto', padding: '1.5rem', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'}}>
      <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1.5rem'}}>Document Upload</h2>
      <form onSubmit={handleSubmit}>
        {['proposal', 'financial_statement', 'operations_license'].map((docType) => (
          <div key={docType} style={{marginBottom: '1rem'}}>
            <label style={{display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem'}}>
              {docType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              {docType !== 'operations_license' && ' *'}
            </label>
            <div
              {...getRootProps()}
              style={{
                border: '2px dashed #D1D5DB',
                borderRadius: '0.375rem',
                padding: '1rem',
                backgroundColor: isDragActive ? '#EFF6FF' : 'transparent',
                cursor: 'pointer'
              }}
            >
              <input {...getInputProps({ name: docType })} />
              {files[docType] ? (
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <span style={{fontSize: '0.875rem'}}>{files[docType].name}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFiles((prevFiles) => ({ ...prevFiles, [docType]: null }));
                    }}
                    style={{background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem'}}
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div style={{textAlign: 'center'}}>
                  <div style={{fontSize: '2rem', color: '#9CA3AF', marginBottom: '0.5rem'}}>⇧</div>
                  <p style={{fontSize: '0.875rem', color: '#6B7280'}}>
                    Drag &amp; drop or click to select a file
                  </p>
                </div>
              )}
            </div>
            {errors[docType] && (
              <div style={{marginTop: '0.5rem', padding: '0.5rem', backgroundColor: '#FEE2E2', borderRadius: '0.375rem', color: '#DC2626'}}>
                <p style={{fontSize: '0.875rem'}}>{errors[docType]}</p>
              </div>
            )}
          </div>
        ))}
        <button 
          type="submit" 
          style={{
            width: '100%', 
            padding: '0.5rem 1rem', 
            backgroundColor: '#3B82F6', 
            color: 'white', 
            borderRadius: '0.375rem', 
            border: 'none', 
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          Upload Documents
        </button>
      </form>
    </div>
  );
};

export default DocumentUpload;