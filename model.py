from typing import List

import torch
from PIL import Image

from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_extractor = ViTFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-125M")
# tokenizer.pad_token = tokenizer.eos_token
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def forward(model: VisionEncoderDecoderModel, images: List[Image.Image], texts: List[str]):
	pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
	pixel_values = pixel_values.to(device)

	tokens = tokenizer(texts, padding=True, return_tensors="pt")
	input_ids = tokens['input_ids'].to(device)

	return model.forward(
		pixel_values=pixel_values,
		# Automatically figures out the decoder input_ids
		labels=input_ids,
	)

def generate(model: VisionEncoderDecoderModel, image: Image.Image, prompt: str, do_sample=True, temperature=0.1, **kwargs):
	pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
	pixel_values = pixel_values.to(device)

	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
	# `model.generate` for encoder-decoder models by default will only accept the encoder inputs.
	# to provide decoder inputs, use forced_decoder_ids.
	# forced_decoder_ids = [[i, input_ids[i]] for i in range(len(input_ids))]

	# Modified to accept decoder input ids
	outputs = model.generate(
		pixel_values=pixel_values,
		decoder_input_ids=input_ids,
		do_sample=do_sample,
		temperature=temperature,
		max_new_tokens=20,
		min_length=input_ids.shape[1] + 1,
		**kwargs,
	)
	prediction = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
	return prediction
