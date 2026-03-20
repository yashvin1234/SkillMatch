import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/UploadForm.css';
 
const UploadForm: React.FC = () => {
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
 
  const apiUrl = import.meta.env.VITE_API_URL;
 
  const uploadFile = async (endpoint: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
 
 
    return axios.post(`${apiUrl}/${endpoint}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  };

 
  const handleProcess = async () => {
    if (!jdFile || !resumeFile) {
      alert('Please upload both JD and Resume!');
      return;
    }
    setLoading(true);
    try {
      await uploadFile('jd/upload/jd_resume_match', jdFile);
      await uploadFile('resume/upload/', resumeFile);
 
      const response = await axios.get(`${apiUrl}/process/process/jd_resume_match`);
      const resultData = typeof response.data === 'string' ? JSON.parse(response.data) : response.data;
      navigate('/result', { state: { resultData } });
    } catch (error: any) {
      alert(error.response?.data?.error || 'Error processing files.');
    } finally {
      setLoading(false);
    }
  };
 
  return (
<div className="upload-container">
<div className="upload-card">
<h2 className="upload-title">Analyze Employee Resume Against Project JD</h2>
<p style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#7a869a' }}>
          Upload an employee’s resume and the corresponding project JD to get a detailed AI-powered skill fit analysis.
 
        </p>
<div className="file-uploads">
<div className="file-upload-box">
<label className="file-label">Resume</label>
<input
              type="file"
              accept=".pdf,.docx"
              onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
            />
            {resumeFile && (
<div className="uploaded-file-name">{resumeFile.name}</div>
            )}
</div>
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
          {loading ? 'Processing...' : 'Analyze Now'}
</button>
</div>
</div>
  );
};
 
export default UploadForm;
