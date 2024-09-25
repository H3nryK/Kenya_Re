import boto3

def extract_text_from_image_aws(image_path):
    client = boto3.client('textract')

    with open(image_path, 'rb') as document:
        image_bytes = document.read()

    response = client.detect_document_text(Document={'Bytes': image_bytes})

    # Extracting text from the response
    extracted_text = ""
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + "\n"

    return extracted_text

# Example usage
extracted_text = extract_text_from_image_aws("image.jpg")
print(extracted_text)
