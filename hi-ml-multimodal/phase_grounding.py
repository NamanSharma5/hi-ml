## FIRST ACTIVATE THIS VENV: source ~/../../vol/bitbucket/nns20/hi-ml-up-to-date-venv/bin/activate
import os
import requests

from typing import List
from typing import Tuple

import tempfile
from pathlib import Path

import torch
# from IPython.display import display
# from IPython.display import Markdown

cache_dir = "/vol/biomedic3/bglocker/ugproj2324/nns20/hi-ml/.cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine



text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)
print(f"Using device: {device}")

TypeBox = Tuple[float, float, float, float]

def download_image(image_url: str) -> Path:
    folder_name = ".img"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Specify the path to save the image
    image_path = os.path.join(folder_name, image_url.split("/")[-1])


    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, 'wb') as f:
                f.write(response.content)
    else:
        print(f"Error: Failed to download image. Status code: {response.status_code}")

    # return Path object
    return Path(image_path)

def plot_phrase_grounding(image_path: Path, text_prompt: str, bboxes: List[TypeBox]) -> None:
    similarity_map = image_text_inference.get_similarity_map_from_raw_data(
        image_path=image_path,
        query_text=text_prompt,
        interpolation="bilinear",
    )
    plot = plot_phrase_grounding_similarity_map(
        image_path=image_path,
        similarity_map=similarity_map,
        bboxes=bboxes
    )
    return plot


def plot_phrase_grounding_from_url(image_url: str, text_prompt: str, bboxes: List[TypeBox]) -> None:
    image_path = download_image(image_url)
    print(f"{image_path}")
    return plot_phrase_grounding(image_path, text_prompt, bboxes)

# image_url = "https://openi.nlm.nih.gov/imgs/512/242/1445/CXR1445_IM-0287-4004.png"
# text_prompt = "Left basilar consolidation seen"

image_url = "https://openi.nlm.nih.gov/imgs/512/246/3833405/PMC3833405_CRIM.ANESTHESIOLOGY2013-524348.001.png"
text_prompt = "three right chest tubes"

# Ground-truth bounding box annotation(s) for the input text prompt
bboxes = [
    (306, 168, 124, 101),
]

text = (
    'The ground-truth bounding box annotation for the phrase'
    f' *{text_prompt}* is shown in the middle figure (in black).'
)

print(text)
plot_phrase_grounding_from_url(image_url, text_prompt, bboxes)
print("Done!")

