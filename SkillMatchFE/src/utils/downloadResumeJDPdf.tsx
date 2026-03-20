import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

// Define the type for course card objects
export type CourseRecommendation = {
  skillArea: string;
  topic: string;
  duration: string;
  url: string;
  objective: string;
};

export async function downloadResultWithCourseLinksPDF(
  resultRef: HTMLElement,
  courseRecommendations: CourseRecommendation[],
  fileName: string = 'analysis-result.pdf'
) {
  const canvas = await html2canvas(resultRef, { scale: 2 });
  const imgData = canvas.toDataURL('image/jpeg', 1);

  const pdf = new jsPDF({
    orientation: 'p',
    unit: 'pt',
    format: [canvas.width, canvas.height],
  });
  pdf.addImage(imgData, 'JPEG', 0, 0, canvas.width, canvas.height);

  const courseButtons = resultRef.querySelectorAll('.course-card .course-link');
  courseButtons.forEach((btn: Element, i: number) => {
    const rect = (btn as HTMLElement).getBoundingClientRect();
    const parentRect = resultRef.getBoundingClientRect();
    const scale = 2;
    const x = (rect.left - parentRect.left) * scale;
    const y = (rect.top - parentRect.top) * scale;
    const w = rect.width * scale;
    const h = rect.height * scale;

    const url = courseRecommendations[i]?.url || '#';
    if (url && url !== '#') {
      pdf.link(x, y, w, h, { url });
    }
  });

  pdf.save(fileName);
}