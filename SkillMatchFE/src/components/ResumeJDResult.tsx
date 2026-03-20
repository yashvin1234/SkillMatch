import React, { useRef, useState} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { downloadResultWithCourseLinksPDF } from '../utils/downloadResumeJDPdf'
import "../styles/Result.css";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";


// InfoCard UI
type InfoCardProps = {
  icon: React.ReactNode;
  title: string;
  color?: string;
  children: React.ReactNode;
  className?: string;
};

const InfoCard: React.FC<InfoCardProps> = ({
  icon,
  title,
  color,
  children,
  className,
}) => (
  <div
    className={`info-card${className ? " " + className : ""}`}
    style={color ? { borderLeft: `5px solid ${color}` } : {}}
  >
    <div className="info-card-title">
      <span className="icon">{icon}</span>
      <span>{title}</span>
    </div>
    {children}
  </div>
);

// Course card UI
type CourseRecommendation = {
  skillArea: string;
  topic: string;
  duration: string;
  url: string;
  objective: string;
};

const CourseCard: React.FC<{ course: CourseRecommendation }> = ({ course }) => (
  <div className="course-card">
    <div className="course-header">
      <div className="skill-area">{course.skillArea}</div>
      <div className="course-topic">{course.topic}</div>
    </div>
    <div className="course-details">
      <div>
        <strong>Duration:</strong> {course.duration}
      </div>
      <div>
        <strong>Objective:</strong>{" "}
        <span style={{ whiteSpace: "pre-wrap" }}>{course.objective}</span>
      </div>
    </div>
    <a
      href={course.url || "#"}
      target="_blank"
      rel="noopener noreferrer"
      className="course-link"
    >
      Go to Course
    </a>
  </div>
);

// Main result UI
const ResumeJDResult: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const resultRef = useRef<HTMLDivElement>(null);
  const [showTechnical, setShowTechnical] = useState(false);
  const [forceShowTechnical, setForceShowTechnical] = useState(false);
  const [pdfMode, setPdfMode] = useState(false);
  const resultData = location.state?.resultData || {};
  const resumeFileName = resultData.Resume_Filename || "analysis-result";
  const handleBackToHome = () => navigate("/");

  const handleDownloadPDF = () => {
    setPdfMode(true);
    setForceShowTechnical(true);
    setTimeout(async () => {
      if (resultRef.current) {
        const pdfFileName = `${resumeFileName} Result Analysis.pdf`;
        await downloadResultWithCourseLinksPDF(
          resultRef.current,
          courseRecommendations,
          pdfFileName  
        );
      }
      setTimeout(() => {
        setForceShowTechnical(false);
        setPdfMode(false);
      }, 500);
    }, 0);
  };



  if (!resultData) {
    return (
      <div className="result-container">
        <div className="result-card">
          <h2 className="analysis-title">No Results Available</h2>
          <div className="action-buttons">
            <button className="secondary-btn" onClick={handleBackToHome}>
              Analyze Another Pair
            </button>
          </div>
        </div>
      </div>
    );
  }


  const parseScore = (scoreStr: string) => {
    if (!scoreStr) return 0;
    const match = scoreStr.match(/^(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  };
  const rawScore = parseScore(resultData.JD_MatchScore);
  const scorePercent = Math.min(100, Math.max(0, Math.round((rawScore / 10) * 100)));
  let ringColor = "#f87171";
  let label = "Low Compatibility";
  if (rawScore >= 7.5) {
    ringColor = "#22c55e";
    label = "High Compatibility";
  } else if (rawScore >= 5) {
    ringColor = "#facc15";
    label = "Moderate Compatibility";
  }

  const gapsDetected =
    Array.isArray(resultData.Key_Gaps) && resultData.Key_Gaps.length > 0;
  const gapsPercent = gapsDetected ? 100 : 0;
  const gapsColor = "#ffbd2f";
  const gapsLabel = gapsDetected ? "Key Gaps Detected!" : "No Key Gaps";

  const hasGrammarErrors =
    (Array.isArray(resultData.Grammatical_Errors) &&
      resultData.Grammatical_Errors.filter((e: string) => !!e && !!e.trim())
        .length > 0) ||
    (Array.isArray(resultData.Spelling_Mistakes) &&
      resultData.Spelling_Mistakes.filter((e: string) => !!e && !!e.trim())
        .length > 0);

  const grammarColor = hasGrammarErrors ? "#e43838" : "#38e44c";
  const grammarLabel = hasGrammarErrors ? "Grammar Error!" : "No Grammar Error";

  const suggestedQuestions: string[] = Array.isArray(resultData.Suggested_Questions)
    ? resultData.Suggested_Questions
    : [];


  // Dynamic course recommendations extraction with URL parsing from multiline course string
  const courseRecommendations: CourseRecommendation[] = Array.isArray(resultData.Suggest_course)
    ? resultData.Suggest_course.map((c: any) => {
      const lines = c.course.split("\n").map((l: string) => l.trim());

      const skillAreaLine = lines.find((line: string) =>
        line.toLowerCase().startsWith("skill area:")
      );
      const subSkillLine = lines.find((line: string) =>
        line.toLowerCase().startsWith("sub-skill:")
      );
      const durationLine = lines.find((line: string) =>
        line.toLowerCase().startsWith("duration:")
      );
      const learningObjectivesLine = lines.find((line: string) =>
        line.toLowerCase().startsWith("learning objectives:")
      );
      const recommendedResourceLineIndex = lines.findIndex((line: string) =>
        line.toLowerCase().startsWith("recommended resource:")
      );

      // Extract URL from recommended resource line or subsequent lines
      let url = "#";
      if (recommendedResourceLineIndex !== -1) {
        const resLine = lines[recommendedResourceLineIndex];
        const urlMatch = resLine.match(/(https?:\/\/[^\s]+)/);
        if (urlMatch) {
          url = urlMatch[0];
        } else if (lines.length > recommendedResourceLineIndex + 1) {
          for (let i = recommendedResourceLineIndex + 1; i < lines.length; i++) {
            const possibleUrl = lines[i];
            if (possibleUrl.startsWith("http://") || possibleUrl.startsWith("https://")) {
              url = possibleUrl;
              break;
            }
          }
        }
      } else {
        // fallback find first url in any line
        for (const line of lines) {
          const uMatch = line.match(/(https?:\/\/[^\s]+)/);
          if (uMatch) {
            url = uMatch[0];
            break;
          }
        }
      }

      return {
        skillArea: skillAreaLine ? skillAreaLine.split(":")[1].trim() : "N/A",
        topic: subSkillLine ? subSkillLine.split(":")[1].trim() : "N/A",
        duration: durationLine ? durationLine.split(":")[1].trim() : "N/A",
        url,
        objective: learningObjectivesLine
          ? learningObjectivesLine.split(":")[1].trim()
          : "No learning objectives provided.",
      };
    })
    : [];

  return (
    <div className="result-container">
      <div className="result-card-wide" ref={resultRef}>
        <div className="result-top-header">
          <div className={`analysis-title${pdfMode ? " pdf-mode" : ""}`}>
            Analysis Results
          </div>
          <div className="analysis-subtitle">
            Detailed breakdown of skill match, gaps, and overall fit for the chosen assignment.
          </div>
        </div>

        {/* Circles Row */}
        <div className="grid-summary-row">
          <InfoCard icon="💼" title="Compatibility Score" color="#2196f3">
            <div className="circle-summary small-circle-summary">
              <CircularProgressbar
                value={scorePercent}
                text={`${rawScore}/10`}
                styles={buildStyles({
                  textColor: "#1e293b",
                  pathColor: ringColor,
                  trailColor: "#e0e6ed",
                  textSize: "1.35rem",
                  pathTransitionDuration: 0.7,
                })}
                strokeWidth={15}
              />
              <div
                className="score-label small-score-label"
                style={{ color: ringColor, fontWeight: 600 }}
              >
                {label}
              </div>
            </div>
          </InfoCard>
          <InfoCard icon="🛑" title="Skill Gaps" color="#ffbd2f">
            <div className="circle-summary small-circle-summary">
              <CircularProgressbar
                value={gapsPercent}
                text={gapsDetected ? "!" : "✓"}
                styles={buildStyles({
                  textColor: "#1e293b",
                  pathColor: gapsColor,
                  trailColor: "#e0e6ed",
                  textSize: "1.1rem",
                  pathTransitionDuration: 0.7,
                })}
                strokeWidth={9}
              />
              <div
                className="score-label small-score-label"
                style={{ color: "#b38113", fontWeight: 600 }}
              >
                {gapsLabel}
              </div>
            </div>
          </InfoCard>
          <InfoCard icon="✅" title="Grammar" color={grammarColor}>
            <div className="circle-summary small-circle-summary">
              <CircularProgressbar
                value={100}
                text={hasGrammarErrors ? "!" : "✓"}
                styles={buildStyles({
                  textColor: "#1e293b",
                  pathColor: grammarColor,
                  trailColor: "#e0e6ed",
                  textSize: "1.1rem",
                  pathTransitionDuration: 0.7,
                })}
                strokeWidth={9}
              />
              <div
                className="score-label small-score-label"
                style={{ color: grammarColor, fontWeight: 600 }}
              >
                {grammarLabel}
              </div>
            </div>
          </InfoCard>
        </div>

        {/* Score Explanation */}
        <InfoCard icon="📝" title="Score Explanation" color="#1e90ff">
          <div style={{ marginBottom: "0.5rem", fontWeight: 500 }}>
            {resultData.Score_Explanation_NonTechnical ||
              "No non-technical explanation available."}
          </div>
          <button
            className="show-details-btn"
            onClick={() => setShowTechnical(!showTechnical)}
            aria-expanded={showTechnical}
            aria-label="Toggle technical explanation"
          >
            {showTechnical ? "Hide Details ▲" : "Technical Details ▼"}
          </button>
          {(showTechnical || forceShowTechnical) && (
            <div style={{ marginTop: "0.5rem", color: "#475569", whiteSpace: "pre-wrap" }}>
              {resultData.Score_Explanation_Technical ||
                "No technical explanation available."}
            </div>
          )}
        </InfoCard>

        {/* Key Matches & Gaps */}
        <div className="key-cards-row">
          <InfoCard icon="🔑" title="Key Matches" color="#22c55e" className="flex-child">
            {Array.isArray(resultData.Key_Matches) && resultData.Key_Matches.length ? (
              <ul className="details-list success">
                {resultData.Key_Matches.map((m: string, idx: number) => (
                  <li key={idx}>
                    <span className="match-icon">✔️</span>
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">No strong matches detected.</p>
            )}
          </InfoCard>
          <InfoCard icon="⚠️" title="Key Gaps" color="#ffbd2f" className="flex-child">
            {Array.isArray(resultData.Key_Gaps) && resultData.Key_Gaps.length ? (
              <ul className="details-list warning">
                {resultData.Key_Gaps.map((g: string, idx: number) => (
                  <li key={idx}>
                    <span className="warn-icon">⚠️</span>
                    <span>{g}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">No major gaps detected.</p>
            )}
          </InfoCard>
        </div>

        {/* Grammar, Spelling, Recommendations */}
        <div className="main-grid-ui">
          <div>
            <InfoCard icon="🔡" title="Grammar Check" color="#38e44c">
              {Array.isArray(resultData.Grammatical_Errors) &&
                resultData.Grammatical_Errors.length ? (
                <ul className="details-list warning">
                  {resultData.Grammatical_Errors.map((e: string, i: number) => (
                    <li key={i}>
                      <span className="warn-icon">⚠️</span> {e}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">No grammatical errors detected.</p>
              )}
            </InfoCard>
            <InfoCard icon="📝" title="Spelling Mistakes" color="#38e44c">
              {Array.isArray(resultData.Spelling_Mistakes) &&
                resultData.Spelling_Mistakes.length ? (
                <ul className="details-list warning">
                  {resultData.Spelling_Mistakes.map((e: string, i: number) => (
                    <li key={i}>
                      <span className="warn-icon">⚠️</span> {e}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">No spelling mistakes detected.</p>
              )}
            </InfoCard>
            <InfoCard icon="🏢" title="Client/Organization Names" color="#5778f8">
              {Array.isArray(resultData.Client_Names) &&
                resultData.Client_Names.length ? (
                <ul className="details-list neutral">
                  {resultData.Client_Names.map((e: string, i: number) => (
                    <li key={i}>
                      <span className="info-icon">🏢</span> {e}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">No confidential client names detected.</p>
              )}
            </InfoCard>
          </div>
          <div>
            <InfoCard icon="💡" title="Recommendations" color="#209ffc">
              {Array.isArray(resultData.Recommendations) &&
                resultData.Recommendations.length ? (
                <ul className="details-list info">
                  {resultData.Recommendations.map((r: string, idx: number) => (
                    <li key={idx}>
                      <span className="info-icon">ℹ️</span>
                      <span>{r}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">No recommendations needed.</p>
              )}
            </InfoCard>
          </div>
        </div>

        {/* Suggested Questions */}
        <div className="suggested-questions-wide" style={{ marginTop: "2rem" }}>
          <InfoCard icon="🎤" title="Suggested Interview Questions" color="#5f76ff">
            {suggestedQuestions.length ? (
              <ul className="details-list">
                {suggestedQuestions.map((q: string, idx: number) => (
                  <li key={idx}>
                    <span style={{ marginRight: 7 }}>❓</span>
                    <span>{q}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">No questions found.</p>
            )}
          </InfoCard>
        </div>

        {/* Course Recommendations */}
        <div className="course-recommendations-section" style={{ marginTop: "2rem" }}>
          <InfoCard icon="🎓" title="Course Recommendations" color="#3bb273">
            <div className="course-cards-row">
              {
                courseRecommendations.length > 0 ? (
                  courseRecommendations
                    .filter((course) => course.url && course.url !== "#" && course.url.trim() !== "")
                    .length > 0 ? (
                    courseRecommendations
                      .filter((course) => course.url && course.url !== "#" && course.url.trim() !== "")
                      .map((course, idx) => <CourseCard key={idx} course={course} />)
                  ) : (
                    <p className="empty-state">No valid course links available.</p>
                  )
                ) : (
                  <p className="empty-state">No course recommendations available.</p>
                )
              }


            </div>
          </InfoCard>
        </div>

        {/* Action Buttons */}
        {!pdfMode && (
          <div className="action-buttons" style={{ marginTop: 32 }}>
            <button className="primary-btn" onClick={handleDownloadPDF}>
              Download as PDF
            </button>
            <button className="secondary-btn" onClick={handleBackToHome}>
              Home
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResumeJDResult;
