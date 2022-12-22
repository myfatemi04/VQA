import json
from typing import List

import numpy as np
import torch
import torch.utils.data
import tqdm
from PIL import Image
from transformers import (AutoTokenizer, CLIPVisionModel, GPT2Model,
                          GPT2Tokenizer, VisionEncoderDecoderModel,
                          ViTImageProcessor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor: ViTImageProcessor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def collate_fn(batch):
	images, questions, answers = zip(*batch)
	return images, questions, answers

class VQADataset(torch.utils.data.Dataset):
	def __init__(self, images_path, split_name, questions, annotations):
		super().__init__()

		self.images_path = images_path
		# train2014 or val2014
		self.split_name = split_name
		self.questions = questions
		self.annotations = annotations

	def __len__(self):
		return len(self.questions['questions'])

	def __getitem__(self, index):
		question = self.questions['questions'][index]
		annotation = self.annotations['annotations'][index]
		question_text = question['question']
		answer_text = annotation['answers'][0]['answer']
		image_id = question['image_id']
		image = Image.open(f"{self.images_path}/COCO_{self.split_name}2014_{image_id:0>12}.jpg")
		if image.mode != "RGB":
			image = image.convert(mode="RGB")
		
		return image, question_text, answer_text

def calculate_loss(model: VisionEncoderDecoderModel, images: List[Image.Image], text_batch: str):
	pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
	pixel_values = pixel_values.to(device)

	tokens = tokenizer(text_batch, padding=True, return_tensors="pt")
	input_ids = tokens['input_ids'].to(device)

	outputs = model.forward(
		pixel_values=pixel_values,
		# Automatically figures out the decoder input_ids
		labels=input_ids,
	)
	loss = outputs.loss
	return loss

def load_split(split_name):
	with open(f"{split_name}/v2_mscoco_{split_name}2014_annotations.json") as f:
		annotations = json.load(f)
	with open(f"{split_name}/v2_OpenEnded_mscoco_{split_name}2014_questions.json") as f:
		questions = json.load(f)
	return VQADataset(f'{split_name}/images', split_name, questions, annotations)

def predict(model: VisionEncoderDecoderModel, image: Image.Image, question: str):
	pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
	pixel_values = pixel_values.to(device)

	input_ids = tokenizer(f"Answer the following question: {question}\nAnswer:", return_tensors="pt").input_ids.to(device)
	# `model.generate` for encoder-decoder models by default will only accept the encoder inputs.
	# to provide decoder inputs, use forced_decoder_ids.
	# forced_decoder_ids = [[i, input_ids[i]] for i in range(len(input_ids))]

	# Modified to accept decoder input ids
	outputs = model.generate(
		pixel_values=pixel_values,
		decoder_input_ids=input_ids,
		max_length=100,
		do_sample=True,
		temperature=0.1,
	)
	prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
	return prediction

def run_training():
	# clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
	# vit = clip.vision_model
	# gpt2 = GPT2Model.from_pretrained("gpt2-large")
	# model = VisionEncoderDecoderModel(encoder=vit, decoder=gpt2)
	model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	train_dataset = load_split('train')
	val_dataset = load_split('val')

	batch_size = 8

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
	# Cosine annealing
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
	for epoch in range(25):
		epoch_total_loss = 0
		epoch_total_n = 0
		with tqdm.tqdm(total=len(train_dataset), desc='epoch ' + str(epoch)) as pbar:
			for batch in train_dataloader:
				images, questions, answers = batch
				texts = [f"Answer the following question: {question}\nAnswer: {answer}" for question, answer in zip(questions, answers)]
				loss = calculate_loss(model, images, texts)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				epoch_total_loss += loss.item() * len(images)
				epoch_total_n += len(images)
				pbar.update(len(images))
				pbar.set_postfix(loss=loss.item(), epoch_loss=epoch_total_loss / epoch_total_n)
		torch.save(model.state_dict(), f"model_{epoch}.pt")
	
	# Demonstrate results
	# For now, overfit and show the results on the training data
	for batch in train_dataloader:
		images, questions, answers = batch
		for image, question, answer in zip(images, questions, answers):
			prediction = predict(model, image, question)
			print(f"Question: {question}")
			print(f"Answer: {answer}")
			print(f"Prediction: {prediction}")
			print()
		break

def run_validation():
	model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	model.load_state_dict(torch.load("vqa_model_epoch_0.pt", map_location=device))
	model.to(device)

	batch_size = 32

	val_dataset = load_split('val')
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	# Demonstrate results
	for batch in val_dataloader:
		images, questions, answers = batch
		for image, question, answer in zip(images, questions, answers):
			prediction = predict(model, image, question)
			print(f"Question: {question}")
			print(f"Answer: {answer}")
			print(f"Prediction: {prediction}")
			print()
		break

if __name__ == '__main__':
	run_validation()
