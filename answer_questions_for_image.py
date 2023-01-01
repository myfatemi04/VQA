import torch
import torch.utils.data
import tqdm
import wandb
from model import device, forward, generate
from vqa_dataset import collate_fn, load_split

from transformers import VisionEncoderDecoderModel


def run_training():
	wandb.init('train', dir='runs', project='vqa', name='imitation-learning')
	wandb.config.encoder = 'openai/clip-vit-large-patch14'
	wandb.config.decoder = 'EleutherAI/gpt-neo-125M'
	wandb.define_metric("batch")
	wandb.define_metric("epoch")
	wandb.define_metric("train_loss", step_metric="batch")
	wandb.define_metric("val_loss", step_metric="epoch")

	model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	model.to(device)

	train_dataset = load_split('train')
	val_dataset = load_split('val')

	batch_size = 8

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
	# Cosine annealing
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

	batch_step = 0
	log_period = 100
	for epoch_step in range(25):
		running_loss = 0

		with tqdm.tqdm(total=len(train_dataset), desc='epoch ' + str(epoch_step)) as pbar:
			for (images, questions, answers) in train_dataloader:
				batch_step += 1

				texts = [f"Answer the following question: {question}\nAnswer: {answer}" for question, answer in zip(questions, answers)]

				optimizer.zero_grad()
				loss = forward(model, images, texts).loss
				loss.backward()
				optimizer.step()
				scheduler.step()

				pbar.update(batch_size)

				running_loss += loss.item()
				if batch_step % log_period == 0:
					wandb.log({'train_loss': running_loss / log_period, 'batch': batch_step})
					pbar.set_postfix({'train_loss': running_loss / log_period})
					running_loss = 0
		
		# Validation
		with torch.no_grad():
			running_loss = 0
			for batch in val_dataloader:
				loss = forward(model, *batch).loss
				running_loss += loss.item()

			wandb.log({'val_loss': running_loss / len(val_dataset), 'epoch': epoch_step})
			print(f"Validation loss: {running_loss / len(val_dataset)}")

		torch.save(model.state_dict(), f"model_{epoch_step}.pt")
	
	# Demonstrate results
	# For now, overfit and show the results on the training data
	for batch in train_dataloader:
		images, questions, answers = batch
		for image, question, answer in zip(images, questions, answers):
			prediction = generate(model, image, question)
			print(f"Question: {question}")
			print(f"Answer: {answer}")
			print(f"Prediction: {prediction}")
			print()
		break

def run_validation():
	import matplotlib.pyplot as plt

	model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	model.load_state_dict(torch.load("vqa_model_epoch_0.pt", map_location=device))
	model.to(device)

	batch_size = 8

	val_dataset = load_split('val')
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

	# Demonstrate results
	for batch in val_dataloader:
		images, questions, answers = batch
		for image, question, answer in zip(images, questions, answers):
			prediction = generate(
				model,
				image,
				f'Answer the following question:', # {question}\nAnswer: ',
				num_return_sequences=5,
				temperature=1
			)
			print(f"Question: {question}")
			print(f"Answer: {answer}")
			print(f"Predictions: {prediction}")
			print()
			plt.imshow(image)
			plt.show()
		break

if __name__ == '__main__':
	run_validation()
	# run_training()
