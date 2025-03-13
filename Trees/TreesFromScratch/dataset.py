from pathlib import Path
import requests
import typer
from loguru import logger
from tqdm import tqdm
import os
from dotenv import load_dotenv

from TreesFromScratch.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
import base64



@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "train.csv",
):
    logger.info("Processing train.csv dataset...")

    # 





    # f=open(output_path, 'wb')
    # for chunk in r.iter_content(chunk_size = 512 * 1024):
    #     if chunk:
    #         f.write(chunk)

    logger.success("Processing dataset complete.")

    


if __name__ == "__main__":
    app()


# base_url = "https://www.kaggle.com/c"
# owner_slug = "kaggle"
# dataset_slug = "spaceship-titanic"
# dataset_version = "1"
# data_url = url = f"{base_url}/{dataset_slug}/download/train.csv"

# dest_path = '../data/raw/dataset.csv'
# kaggle_info = {'UserName': os.getenv('kaggleuser'), 'Password': os.getenv('kagglepwd'), 'key': os.getenv('key')}

# creds = base64.b64encode(bytes(f"{kaggle_info['UserName']}:{kaggle_info['Password']}", "ISO-8859-1")).decode("ascii")
# headers = {
# "Authorization": f"Basic {creds}"
# }

# r = requests.get(data_url, headers=headers)
# assert(r.status_code == 200)

# f=open(dest_path, 'wb')
# for chunk in r.iter_content(chunk_size = 512 * 1024):
#     if chunk:
#         f.write(chunk)