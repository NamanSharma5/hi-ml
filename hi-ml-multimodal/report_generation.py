
from transformers import BertLMHeadModel, AutoModelForCausalLM

import os
cache_dir = "/vol/biomedic3/bglocker/ugproj2324/nns20/hi-ml/.cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir


import torch
from health_multimodal.image.model.pretrained import (
    BIOMED_VLP_BIOVIL_T,
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    BIOVIL_T_COMMIT_TAG,
    CXR_BERT_COMMIT_TAG,
)
from health_multimodal.text.model import CXRBertTokenizer

model_identifier = BIOMED_VLP_BIOVIL_T
# model_identifier = BIOMED_VLP_CXR_BERT_SPECIALIZED

tokenizer = CXRBertTokenizer.from_pretrained(model_identifier, revision=BIOVIL_T_COMMIT_TAG)  
# model = BertLMHeadModel.from_pretrained(model_identifier, is_decoder=True)
model = AutoModelForCausalLM.from_pretrained(model_identifier, is_decoder=True)

# Function to encode input text and decode output text
def generate_text(input_prompt, model, tokenizer, max_length=50):
    # Encode the input prompt to get token ids
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

    # Decoding loop
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=0.85,
        do_sample=True,
        top_k=125,
        early_stopping=True
    )

    # Decode to text
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

input_prompt = "The finding suggests"
# input_prompt = "What does it mean when doctors say there is a consolidation in a chest x-ray?"

output_text = generate_text(input_prompt, model, tokenizer)
print(output_text)



