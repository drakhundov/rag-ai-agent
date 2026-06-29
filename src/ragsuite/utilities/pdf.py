# ! requires pymupdf pillow pytesseract pypdf
from io import BytesIO

import fitz
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter


# ! Writes to the directory instead of returning binary.
def perform_ocr_on_pdf(
    src_path: str, dest_path: str, dpi: int = 300, lang: str = "tur+eng"
) -> None:
    doc = fitz.open(src_path)
    writer = PdfWriter()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf", lang=lang)
        reader = PdfReader(BytesIO(pdf_bytes))
        for p in reader.pages:
            writer.add_page(p)
    with open(dest_path, "wb") as f:
        writer.write(f)
