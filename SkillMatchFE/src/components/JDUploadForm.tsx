import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/UploadForm.css';

const apiUrl = import.meta.env.VITE_API_URL;

const JDUploadForm: React.FC = () => {
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const uploadFile = async (endpoint: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    // return axios.post(`http://localhost:8000/${endpoint}`, formData, {
    //   headers: { 'Content-Type': 'multipart/form-data' },
    // });
    return axios.post(`${apiUrl}/${endpoint}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  };

  const handleProcess = async () => {
    if (!jdFile) {
      alert('Please upload a Job Description!');
      return;
    }
    setLoading(true);
    try {
      await uploadFile('jd/upload/analyze_jd', jdFile);
      // const response = await axios.get('http://localhost:8000/process/process/analyze_jd/');
      const response = await axios.get(`${apiUrl}/process/process/analyze_jd/`);
      const resultData = typeof response.data === 'string' ? JSON.parse(response.data) : response.data;
      navigate('/jd-result', { state: { resultData } });
    } catch (error: any) {
      alert(error.response?.data?.error || 'Error processing file.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-card">
        <h2 className="upload-title">Upload Job Description</h2>
        <p style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#7a869a' }}>
          Let our AI analyze the job description and extract essential details.
        </p>
        <div className="file-uploads">
          <div className="file-upload-box">
            <label className="file-label">Job Description</label>
            <input
              type="file"
              accept=".pdf,.docx"
              onChange={(e) => setJdFile(e.target.files?.[0] || null)}
            />
            {jdFile && (
              <div className="uploaded-file-name">{jdFile.name}</div>
            )}
          </div>
        </div>
        <button
          className="upload-btn"
          onClick={handleProcess}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Analyze JD'}
        </button>
      </div>
    </div>
  );
};

export default JDUploadForm;
