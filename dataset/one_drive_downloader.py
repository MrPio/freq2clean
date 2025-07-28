import requests
from tqdm import tqdm

"""
Please follow the instruction available here: https://stackoverflow.com/a/79714607/19815002
"""

url = "PLACE HERE THE URL"

headers = {
    # PLACE HERE THE HEADERS
}
data = """
PLACE HERE THE BODY PAYLOAD
"""

with requests.post(url, headers=headers, data=data, stream=True) as response:
    if response.ok:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        filename = data.split("=")[1].split("&")[0]

        with open(filename, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data_chunk in response.iter_content(chunk_size=block_size):
                file.write(data_chunk)
                bar.update(len(data_chunk))
        print("✅ Download complete.")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")
        print(response.text)
