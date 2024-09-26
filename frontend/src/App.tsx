import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import logo from './assets/knre.webp';

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

  const Logo = () => (
    <img 
      src={logo} 
      alt="Company Logo" 
      style={{
        width: '100px',
        height: '100px',
        margin: '0 auto 2rem',
        display: 'block',
        borderRadius: '50%',
        objectFit: 'cover'
      }}
    />
  );

  const Dropzone = ({ docType }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop: (acceptedFiles) => onDrop(acceptedFiles, [], { target: { name: docType } }),
    });

    return (
      <div
        {...getRootProps()}
        style={{
          border: '2px dashed #D1D5DB',
          borderRadius: '0.375rem',
          padding: '1rem',
          backgroundColor: isDragActive ? '#EFF6FF' : 'transparent',
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
      >
        <input {...getInputProps({ name: docType })} />
        {files[docType] ? (
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
            <span style={{fontSize: '0.875rem', wordBreak: 'break-all'}}>{files[docType].name}</span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setFiles((prevFiles) => ({ ...prevFiles, [docType]: null }));
              }}
              style={{background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem', transition: 'color 0.3s ease', marginLeft: '8px'}}
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
    );
  };

  if (isSuccess) {
    return (
      <div style={{
        maxWidth: '400px',
        margin: '2rem auto',
        padding: '1.5rem',
        backgroundColor: 'white',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        animation: 'fadeIn 0.5s ease-out'
      }}>
        <Logo />
        <div style={{textAlign: 'center', color: 'green', fontSize: '3rem', marginBottom: '1rem', animation: 'scaleIn 0.5s ease-out'}}>✓</div>
        <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', textAlign: 'center', marginBottom: '1rem'}}>Success!</h2>
        <p style={{textAlign: 'center', marginBottom: '1rem'}}>Your documents have been uploaded successfully.</p>
        <p style={{textAlign: 'center', fontWeight: 'semibold'}}>{quotation}</p>
      </div>
    );
  }

  return (
    <div style={{
      maxWidth: '400px',
      margin: '2rem auto',
      padding: '1.5rem',
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      animation: 'fadeIn 0.5s ease-out'
    }}>
      <Logo />
      <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1.5rem', textAlign: 'center'}}>Document Upload</h2>
      <form onSubmit={handleSubmit}>
        {['proposal', 'financial_statement', 'operations_license'].map((docType) => (
          <div key={docType} style={{marginBottom: '1rem'}}>
            <label style={{display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem'}}>
              {docType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              {docType !== 'operations_license' && ' *'}
            </label>
            <Dropzone docType={docType} />
            {errors[docType] && (
              <div style={{
                marginTop: '0.5rem',
                padding: '0.5rem',
                backgroundColor: '#FEE2E2',
                borderRadius: '0.375rem',
                color: '#DC2626',
                animation: 'shake 0.82s cubic-bezier(.36,.07,.19,.97) both'
              }}>
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
            fontSize: '1rem',
            transition: 'background-color 0.3s ease'
          }}
        >
          Upload Documents
        </button>
      </form>
    </div>
  );
};

const styles = `
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  @keyframes scaleIn {
    from { transform: scale(0); }
    to { transform: scale(1); }
  }
  @keyframes shake {
    10%, 90% { transform: translate3d(-1px, 0, 0); }
    20%, 80% { transform: translate3d(2px, 0, 0); }
    30%, 50%, 70% { transform: translate3d(-4px, 0, 0); }
    40%, 60% { transform: translate3d(4px, 0, 0); }
  }
`;

const DocumentUploadWithStyles = () => (
  <>
    <style>{styles}</style>
    <DocumentUpload />
  </>
);

export default DocumentUploadWithStyles;