import torch
from contextlib import nullcontext
from torch import optim
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

CKPT = "Qwen/Qwen2.5-7B-Instruct"
LOAD_DTYPE = torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM_MESSAGE = (
    "You are a concise tutor helping players learn the expedition glossary. "
    "When a user asks for a translation, answer with the dictionary term only."
)
TEST_QUESTION = "Translate the Kala word 'lurin' using the expedition glossary."
TTT_SAMPLES = [
    {
        "question": "Translate the Kala word 'sori' using the expedition glossary.",
        "answer": "'sori' translates to 'water'.",
    },
    {
        "question": "Translate the Kala word 'maven' using the expedition glossary.",
        "answer": "'maven' translates to 'pathfinder'.",
    },
    {
        "question": "Translate the Kala word 'thrinal' using the expedition glossary.",
        "answer": "'thrinal' translates to 'camp'.",
    },
]


def load_model_and_tokenizer():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = LOAD_DTYPE if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def to_device(batch):
    return {key: value.to(DEVICE) for key, value in batch.items()}


def build_messages(question, answer=None):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def encode_chat(tokenizer, messages, add_generation_prompt):
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )

    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}
    elif hasattr(encoded, "to_dict"):
        encoded = dict(encoded.to_dict())
    else:
        encoded = dict(encoded)

    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return to_device(encoded)


def generate_response(model, tokenizer, question, max_new_tokens=128):
    messages = build_messages(question)
    inputs = encode_chat(tokenizer, messages, add_generation_prompt=True)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    generated_tokens = output[0, inputs["input_ids"].size(-1) :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def mask_prompt_tokens(full_inputs, prompt_length):
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    full_inputs["labels"] = labels
    return full_inputs


def configure_trainable_parameters(model):
    trainable = []
    for name, param in model.named_parameters():
        requires_grad = name.startswith("lm_head")
        param.requires_grad_(requires_grad)
        if requires_grad:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No trainable parameters found on lm_head; check model config.")
    return trainable


def run_test_time_training(model, tokenizer, samples, steps=3, lr=5e-6, grad_clip=0.5):
    model.train()
    model.config.use_cache = False
    trainable = configure_trainable_parameters(model)
    optimizer = optim.AdamW(trainable, lr=lr, eps=1e-8)

    for step in range(steps):
        total_loss = 0.0
        for sample in samples:
            messages = build_messages(sample["question"], sample["answer"])
            full_inputs = encode_chat(tokenizer, messages, add_generation_prompt=False)
            prompt_inputs = encode_chat(tokenizer, messages[:-1], add_generation_prompt=True)
            prompt_length = prompt_inputs["input_ids"].size(-1)
            batch = mask_prompt_tokens(full_inputs, prompt_length)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss).all():
                print("Skipping update due to invalid loss")
                continue
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        average_loss = total_loss / max(1, len(samples))
        print(f"TTT step {step + 1}: average loss = {average_loss:.4f}")

    model.eval()
    model.config.use_cache = True


def main():
    tokenizer, model = load_model_and_tokenizer()
    param_fn = getattr(model, "num_parameters", None)
    num_params = param_fn() if callable(param_fn) else None
    if num_params is not None:
        print(f"Loaded {num_params / 1e6:.0f}M parameters on {DEVICE}.")
    else:
        print("Model loaded; parameter count unavailable.")

    print("\nBaseline response:")
    baseline = generate_response(model, tokenizer, TEST_QUESTION)
    print(baseline)

    print("\nRunning test-time training...")
    run_test_time_training(model, tokenizer, TTT_SAMPLES)

    print("\nAdapted response:")
    adapted = generate_response(model, tokenizer, TEST_QUESTION)
    print(adapted)


if __name__ == "__main__":
    main()
