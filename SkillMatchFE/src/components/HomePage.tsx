import React from 'react';
import { useNavigate } from 'react-router-dom';
import "../styles/HomePage.css";

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/upload');
  };

  const handleAnalyzeJD = () => {
    navigate('/jd-upload');
  };

  return (
    <div className="home-container">
      <h1 className="headline">SkillMatch: Talent Alignment Suite</h1>
      <p className="tagline">Accelerate project staffing by efficiently matching team members’ strengths to project needs.</p>

      {/* Features Boxes */}
      <div className="features-wrapper">
        {/* Box 1 */}
        <div className="feature-box">
          <h2 className="box-title">Resume & JD Analysis</h2>
          <p className="box-desc">
            Upload an employee’s resume together with the role’s job description to receive instant AI analysis on alignment, strengths, and gaps.
          </p>
          <button className="action-btn" onClick={handleGetStarted}>Analyze Now</button>
        </div>
        {/* Box 2 */}
        <div className="feature-box">
          <h2 className="box-title">Job Description Insights</h2>
          <p className="box-desc">
            Extract key requirements and skill benchmarks from any project JD to inform team selection and employee development.
          </p>
          <button className="action-btn" onClick={handleAnalyzeJD}>Upload JD</button>
        </div>
      </div>

      {/* Footer */}
      <footer className="site-footer">&copy; {new Date().getFullYear()} SkillMatch. All rights reserved.</footer>
    </div>
  );
};

export default HomePage;
