import React, { useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { downloadElementAsPDF } from '../utils/downloadJDPdf';
import '../styles/Result.css'

const JDResult: React.FC = () => {
  const locationHook = useLocation();
  const navigate = useNavigate();
  const resultRef = useRef<HTMLDivElement>(null);
  const [pdfMode, setPdfMode] = useState(false);

  const resultData = locationHook.state?.resultData || {};

  const handleDownloadPDF = () => {
    setPdfMode(true);
    setTimeout(() => {
      if (resultRef.current) {
        downloadElementAsPDF(resultRef.current, "jd-analysis-result.pdf");
      }
      setTimeout(() => {
        setPdfMode(false);
      }, 500);
    }, 0);
  };

  const experienceLines = (resultData.experience || "Not specified")
    .split(';')
    .map((item: string) => item.trim())
    .filter((item: string) => item.length > 0);

  return (
    <div className="result-container">
      <div className="result-card-wide" ref={resultRef}>
        <div className="result-top-header">
          <div className={`analysis-title${pdfMode ? " pdf-mode" : ""}`}>JD Analysis Result</div>
          <div className="analysis-subtitle">
            Key insights extracted from your job description.
          </div>
        </div>

        <div className="info-card" style={{ borderLeft: "5px solid #209ffc" }}>
          <div className="info-card-title">
            <span>📝</span> Sanitized Job Description
          </div>
          <div style={{ color: "#3e4251", whiteSpace: 'pre-line' }}>
            {resultData.sanitized_jd || "No content extracted."}
          </div>
        </div>

        <div className="main-grid-ui">
          <div>
            <div className="info-card" style={{ borderLeft: "5px solid #22c55e" }}>
              <div className="info-card-title">
                <span>✅</span> Must-have Skills
              </div>
              <ul className="details-list success">
                {(resultData.must_have_skills || []).map((skill: string, idx: number) => (
                  <li key={idx}><span className="match-icon">✔️</span> {skill}</li>
                ))}
              </ul>
            </div>
            <div className="info-card" style={{ borderLeft: "5px solid #feb247" }}>
              <div className="info-card-title">
                <span>⭐</span> Good-to-have Skills
              </div>
              <ul className="details-list warning">
                {(resultData.good_to_have_skills || []).map((skill: string, idx: number) => (
                  <li key={idx}><span className="warn-icon">⭐</span> {skill}</li>
                ))}
              </ul>
            </div>
          </div>
          <div>
            <div className="info-card" style={{ borderLeft: "5px solid #5778f8" }}>
              <div className="info-card-title">
                <span>📍</span> Location
              </div>
              <div style={{ fontSize: '1.0rem', color: "#3850c1" }}>
                {resultData.location || "Not specified"}
              </div>
            </div>
            <div className="info-card" style={{ borderLeft: "5px solid #c1a855" }}>
              <div className="info-card-title">
                <span>⏳</span> Duration
              </div>
              <div style={{ fontSize: '1.0rem' }}>
                {resultData.duration || "No duration specified"}
              </div>
            </div>
            <div className="info-card" style={{ borderLeft: "5px solid #5da8ff" }}>
              <div className="info-card-title">
                <span>🗓️</span> Experience
              </div>
              <ul style={{ fontSize: '1.0rem', color: "#2367be", marginLeft: 0, paddingLeft: '1rem' }}>
                {experienceLines.length === 0 ? (
                  <li>Not specified</li>
                ) : (
                  experienceLines.map((line: string, idx: number) => (
                    <li key={idx}>{line}</li>
                  ))
                )}
              </ul>
            </div>
          </div>
        </div>
        {!pdfMode && (
          <div className="action-buttons" style={{ marginTop: 32 }}>
            <button className="primary-btn" onClick={handleDownloadPDF}>
              Download as PDF
            </button>
            <button className="secondary-btn" onClick={() => navigate("/")}>
              Home
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default JDResult;
