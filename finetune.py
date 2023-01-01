import torch
from transformers import VisionEncoderDecoderModel
from vqa_dataset import load_split
from model import generate, forward, tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.load_state_dict(torch.load("vqa_model_epoch_0.pt", map_location=device))
model.to(device)

train_dataset = load_split('train')

image, question_text, answer_text = train_dataset[0]
image_id = train_dataset.get_image_id(0)

ratings = []
human_completions = []
num_return_sequences = 5

model.train()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

past_texts = []

for _ in range(64):
    question = input("Question: ")
    context = f"Answer the following question: {question}\nAnswer: "
    completions = generate(model, image, context, num_return_sequences=num_return_sequences, temperature=0.5)
    completions = list({x.strip() for x in completions})
    print("Completions:", completions)

    # print("Rate the completions from 1 to 5 for their relevance and correctness.")
    # for i, completion in enumerate(completions):
    #     print(f"Generation {i + 1}: {completion}")
    #     helpfulness = int(input("Relevance: "))
    #     correctness = int(input("Correctness: "))
    #     coherence = int(input("Coherence: "))
    #     overall = int(input("Overall: "))
    #     ratings.append({
    #         "helpfulness": helpfulness,
    #         "correctness": correctness,
    #         "coherence": coherence,
    #         "overall": overall,
    #         "image_id": image_id,
    #         "context": context,
    #         "completion": completion
    #     })
    
    print("How would you complete the question?")
    print(f"Question: \"{question}\"")
    completion = input("Completion: ")
    human_completions.append({
        "image_id": image_id,
        "context": context,
        "completion": completion
    })

    text = context + completion + tokenizer.eos_token
    past_texts.append(text)
    # for e in range(4):
    #     optim.zero_grad()
    #     loss = forward(model, [image] * min(4, len(past_texts)), past_texts[-4:]).loss
    #     loss.backward()
    #     optim.step()
    #     print(loss.item())

    # completions = generate(model, image, context, num_return_sequences=num_return_sequences, temperature=0.5)
    # completions = list({x.strip() for x in completions})
    # print("After:", completions)

print(ratings)

"""
We are trying to optimize
 * Helpfulness (including relevant information, and excluding irrelevant information)
 * Correctness (including correct information, and excluding incorrect information)
 * Coherence (ensuring that responses make sense)
 * Overall quality
"""
