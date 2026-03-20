import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export async function downloadElementAsPDF(element: HTMLElement, filename = 'document.pdf') {
  

  const canvas = await html2canvas(element, { scale :2 });
  const imgData = canvas.toDataURL('image/jpeg', 1); 
  


  const pdf = new jsPDF({
    orientation: 'p',
    unit: 'pt',
    format: [canvas.width, canvas.height],
  });
  pdf.addImage(imgData, 'JPEG', 0, 0, canvas.width, canvas.height);
  pdf.save(filename);
}


