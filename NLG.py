from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(prompt, max_length=100, temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    prompt = "Once upon a time, in a faraway kingdom"
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)
