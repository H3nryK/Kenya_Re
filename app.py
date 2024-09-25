from google.cloud import vision
import io

def extract_text_from_image_google(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    return texts[0].description if texts else ""

extracted_text = extract_text_from_image_google("image.jpg")
print(extracted_text)