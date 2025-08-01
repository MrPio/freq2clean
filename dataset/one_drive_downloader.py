import requests
from tqdm import tqdm

"""
Please follow the instruction available here: https://stackoverflow.com/a/79714607/19815002
"""

url = "https://westeurope1-mediap.svc.ms/transform/zip?cs=fFNQTw"

headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Type": "application/x-www-form-urlencoded",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "iframe",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-GPC": "1",
        "Priority": "u=4",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
}
data = """
zipFileName=dati_per_erzelli.zip&guid=57389d2d-06ce-448a-9b91-73d3d2105b16&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22dati_per_erzelli%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fistitutoitalianotecnologia-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21YKHUpxIEx0KQXF3flhfvK98JVM_PlHFGr9b2ardyFGA7X2-O65bfRquXWKD0nh9G%2Fitems%2F01PCS3KRCDENRBHKEMCFBI2KT3PTW37GHH%3Fversion%3DPublished%26access_token%3Dv1.eyJzaXRlaWQiOiJhN2Q0YTE2MC0wNDEyLTQyYzctOTA1Yy01ZGRmOTYxN2VmMmIiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaXN0aXR1dG9pdGFsaWFub3RlY25vbG9naWEtbXkuc2hhcmVwb2ludC5jb21AYmYyZjQ2NWUtMWY3NS00YjZmLWJlMjktYmY1MzI3M2JjMWRhIiwiZXhwIjoiMTc1NDAwNjQwMCJ9.CiMKCXNoYXJpbmdpZBIWeDdLaGwrSndwazJ1QXUvbitNUE1aUQoICgNzdHASAXQKCgoEc25pZBICMzMSBgjkvDsQARomMjAwMTpiMDc6NjQ2YjoyODBjOjQ0M2U6ZTJhMDo4NzFmOjRmNWIiFG1pY3Jvc29mdC5zaGFyZXBvaW50KixTcFZ3dzJ2VWxnM3NWN2VXSnVHY09VanRSSmxNcHgybkgvK0E3dWpKSXJFPTCIATgBShBoYXNoZWRwcm9vZnRva2VuYgR0cnVlcj0waC5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhZ3Vlc3QjczExMTg3ODFAc3R1ZGVudGkudW5pdnBtLml0egEwwgE9MCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWd1ZXN0I3MxMTE4NzgxQHN0dWRlbnRpLnVuaXZwbS5pdA.PPc5yJGTqVmigWpT-ThFM1Tcx7IaP_cAA6D4yNShOFg%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken="""

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
