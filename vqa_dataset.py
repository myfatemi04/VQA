import json

import PIL.Image as Image
import torch.utils.data


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

	def get_image_id(self, index):
		return self.questions['questions'][index]['image_id']

def load_split(split_name):
	with open(f"{split_name}/v2_mscoco_{split_name}2014_annotations.json") as f:
		annotations = json.load(f)
	with open(f"{split_name}/v2_OpenEnded_mscoco_{split_name}2014_questions.json") as f:
		questions = json.load(f)

	return VQADataset(f'{split_name}/images', split_name, questions, annotations)
