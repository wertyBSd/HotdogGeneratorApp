import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "better_bing_image_downloader"))
from better_bing_image_downloader.download import downloader

def download_images(query, category_folder, count=100):
    folder = os.path.join("dataset", category_folder)

    # Удаляем папку, если она уже существует
    if os.path.exists(folder):
        shutil.rmtree(folder)

    print(f"\n🔍 Searching for '{query}' images into '{folder}'...")

    downloader(
        query=query,
        limit=count,
        output_dir=folder,
        adult_filter_off=True,
        force_replace=True,
        timeout=60,
        filter="photo",
        verbose=True,
        name=query.replace(" ", "_"),
        max_workers=8
    )

    print(f"\n📁 Finished downloading {count} images for '{query}' into '{folder}'")

# Пример вызова
download_images("cobb salad with bacon and avocado", "not_hotdog", count=100)
