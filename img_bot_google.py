import os
import shutil
import time
import requests
from PIL import Image
from io import BytesIO

# Замените на свои значения
API_KEY = ""
CX = ""

def download_images(query, category_folder, count=50, delay=0.5):
    folder = os.path.join("dataset", category_folder)
    os.makedirs(folder, exist_ok=True)

    print(f"\n🔍 Searching for '{query}' images into '{folder}'...")

    downloaded = 0
    start = 1

    while downloaded < count:
        params = {
            "q": query,
            "searchType": "image",
            "key": API_KEY,
            "cx": CX,
            "num": min(10, count - downloaded),
            "start": start
        }

        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        data = response.json()

        if "items" not in data:
            print("⚠️ Нет результатов или превышен лимит API.")
            break

        for item in data["items"]:
            try:
                url = item["link"]
                img_response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")

                filename = os.path.join(folder, f"{query.replace(' ', '_')}_{downloaded + 1}.jpg")
                img.save(filename, format="JPEG")
                downloaded += 1
                print(f"[{downloaded}/{count}] ✅ Saved: {filename}")
                time.sleep(delay)
            except Exception as e:
                print(f"⚠️ Failed to download image: {e}")

        start += 10

    print(f"\n📁 Finished downloading {downloaded} images for '{query}' into '{folder}'")

