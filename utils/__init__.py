from PIL import Image
import os
import base64
from io import BytesIO
import requests

def get_labels() -> list:
    cur_dir = os.getcwd()
    labels = os.listdir(cur_dir + '/data/Training')
    return labels

def remove_number(label: str) -> str:
    words = label.split()
    words = [word for word in words if not word.isdigit()]
    return ' '.join(words)

def get_image_from_url(url: str):
    """
    Only accepts jpeg and png images or regular URL
    """
    try:
        if 'data:image/jpeg;base64,' in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))
            return img
        elif 'data:image/png;base64,' in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))
            return img
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            return img
    except Exception as e:
        print(e)
        return None

def delete_in_folder(folder: str) -> None:
    """
    Delete all files in a folder
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)
    return None

if __name__ == '__main__':
    print(get_labels())