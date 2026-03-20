import { Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import UploadForm from './components/ResumeJDUploadForm';
import Result from './components/ResumeJDResult';
import JDUploadForm from './components/JDUploadForm'; 
import JDResult from './components/JDAnalysisResult'; 

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/upload" element={<UploadForm />} />
      <Route path="/result" element={<Result />} />
      <Route path="/jd-upload" element={<JDUploadForm />} />   
      <Route path="/jd-result" element={<JDResult />} />       
    </Routes>
  );
}

export default App;