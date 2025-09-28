import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# ============== Setup ==============
CKPT = "Qwen/Qwen2.5-7B-Instruct"        # or "Qwen/Qwen1.5-0.5B"
DTYPE = torch.float16           # bfloat16 also works on A100/H100

tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    torch_dtype=DTYPE,
    device_map={"": "cuda"},    # put everything on GPU-0
    trust_remote_code=True,
)

print(f"Loaded {model.num_parameters()/1e6:.0f} M params")
device = torch.device("cuda")

# ============== TTT Task: Reverse Acronym Expansion ==============
# This task requires learning a specific uncommon expansion pattern from examples

TTT_SAMPLES = [
    {
        "input": "What does NASA stand for in our special coding system?\n\nIn our system:\n- IBM = Intelligent Business Machines\n- FBI = Federal Bureau of Investigation\n- NASA = ?",
        "target": "NASA = Neural Architecture Search Algorithm"
    },
    {
        "input": "Expand the acronym CIA in our special coding system.\n\nIn our system:\n- WHO = World Health Organization\n- CIA = ?",
        "target": "CIA = Computational Intelligence Architecture"
    },
    {
        "input": "What is the expansion of GPU in our special coding system?\n\nIn our system:\n- CPU = Central Processing Unit\n- GPU = ?",
        "target": "GPU = Gradient Propagation Unit"
    },
]

TEST_PROMPT = """What does API stand for in our special coding system?

In our system:
- SQL = Structured Query Language
- API = ?

Based on the pattern, API stands for:"""

# ============== Helper Functions ==============
def prepare_batch(samples, tokenizer, device):
    """Prepare training batch with proper padding and attention masks."""
    texts = []
    for sample in samples:
        text = f"{sample['input']}\n\nAnswer: {sample['target']}"
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Create labels (shift by 1 for next-token prediction)
    labels = inputs.input_ids.clone()

    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100

    return inputs, labels

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0):
    """Generate model response with controlled decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False if temperature == 0 else True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# ============== Baseline Inference ==============
print("\n" + "="*50)
print("BASELINE INFERENCE (Before TTT)")
print("="*50)

baseline_response = generate_response(model, tokenizer, TEST_PROMPT)
print(f"Prompt: {TEST_PROMPT}")
print(f"Baseline Answer: {baseline_response}")

# ============== Test-Time Training ==============
print("\n" + "="*50)
print("TEST-TIME TRAINING")
print("="*50)

# TTT hyperparameters
learning_rate = 5e-5
num_epochs = 3
grad_clip = 1.0

# Prepare optimizer (only optimize the LM head for stability)
trainable_params = [p for p in model.lm_head.parameters()]
optimizer = optim.AdamW(trainable_params, lr=learning_rate)

# Store original weights for potential rollback
original_lm_head = model.lm_head.weight.clone()

model.train()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Prepare batch
    inputs, labels = prepare_batch(TTT_SAMPLES, tokenizer, device)

    # Forward pass
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        labels=labels
    )

    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

    # Update weights
    optimizer.step()

    # Check for NaN and rollback if needed
    if torch.isnan(loss):
        print("NaN detected, rolling back weights...")
        model.lm_head.weight.data = original_lm_head.clone()
        break

# ============== Adapted Inference ==============
print("\n" + "="*50)
print("ADAPTED INFERENCE (After TTT)")
print("="*50)

model.eval()
adapted_response = generate_response(model, tokenizer, TEST_PROMPT)
print(f"Prompt: {TEST_PROMPT}")
print(f"Adapted Answer: {adapted_response}")

print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Before TTT: {baseline_response}")
print(f"After TTT:  {adapted_response}")
print(f"Expected:   Adaptive Programming Interface or similar technical expansion")