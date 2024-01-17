## FIRST ACTIVATE THIS VENV: source ~/../../vol/bitbucket/nns20/hi-ml-mulitmodal-venv/bin/activate

from typing import List
from typing import Tuple

import tempfile
from pathlib import Path

import torch
# from IPython.display import display
# from IPython.display import Markdown

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine



# text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
# image_inference = get_image_inference(ImageModelType.BIOVIL_T)

print("worked")
