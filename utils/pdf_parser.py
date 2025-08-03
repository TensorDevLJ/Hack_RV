import requests
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
def extract_text_from_url(url: str) -> str:
    response = requests.get(url)
    images = convert_from_bytes(response.content)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text