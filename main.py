import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== Setup ==============
CKPT = "Qwen/Qwen2.5-7B-Instruct"        # or "Qwen/Qwen1.5-0.5B"
DTYPE = torch.float16           # bfloat16 also works on A100/H100

tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    torch_dtype=DTYPE,
    device_map={"": "cuda"},    # put everything on GPU-0
    trust_remote_code=True,
)

print(f"Loaded {model.num_parameters()/1e9:.1f}B params")
device = torch.device("cuda")

# ============== TTT Task: Simple Pattern Learning ==============
# Teaching the model a simple substitution cipher pattern

TTT_SAMPLES = [
    "In our code: 'cat' means 'dog' and 'dog' means 'cat'",
    "In our code: 'yes' means 'no' and 'no' means 'yes'",
    "In our code: 'up' means 'down' and 'down' means 'up'",
    "In our code: 'left' means 'right' and 'right' means 'left'"
]

TEST_PROMPT = "In our code: 'hot' means 'cold'. Question: What does 'hot' mean in our code? Answer:"

# ============== Helper Functions ==============
def generate_response(model, tokenizer, prompt, max_new_tokens=20):
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Deterministic
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============== Baseline Inference ==============
print("\n" + "="*50)
print("BASELINE (Before TTT)")
print("="*50)

baseline_response = generate_response(model, tokenizer, TEST_PROMPT)
print(f"Test: {TEST_PROMPT}")
print(f"Model says: {baseline_response}")

# ============== Test-Time Training ==============
print("\n" + "="*50)
print("TEST-TIME TRAINING")
print("="*50)

# Simple training setup
learning_rate = 1e-7  # MUCH smaller - NaNs suggest we're destroying weights
num_steps = 20  # More steps with tiny LR

# Memory-efficient: Only train last few transformer layers
for param in model.parameters():
    param.requires_grad = False

# Enable gradients for last 2 transformer layers + lm_head
for i in range(len(model.model.layers) - 2, len(model.model.layers)):
    for param in model.model.layers[i].parameters():
        param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

# Only optimize parameters that require gradients
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Training {len(trainable_params)} parameter tensors")

# Use SGD instead of Adam - simpler, more stable
optimizer = optim.SGD(trainable_params, lr=learning_rate)

model.train()
for step in range(num_steps):
    for i, sample_text in enumerate(TTT_SAMPLES):
        # Tokenize the full example
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128).to(device)

        # Labels are the same as inputs for language modeling
        labels = inputs.input_ids.clone()

        # Forward pass with mixed precision for stability
        with torch.cuda.amp.autocast(dtype=DTYPE):
            outputs = model(input_ids=inputs.input_ids, labels=labels)
            loss = outputs.loss

        # Skip if loss is already bad
        if torch.isnan(loss) or loss.item() > 10:
            print(f"Step {step+1}.{i+1}, Loss: {loss.item():.4f} - skipping update")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)

        optimizer.step()

        print(f"Step {step+1}.{i+1}, Loss: {loss.item():.4f}")

# ============== Adapted Inference ==============
print("\n" + "="*50)
print("ADAPTED (After TTT)")
print("="*50)

model.eval()
adapted_response = generate_response(model, tokenizer, TEST_PROMPT)
print(f"Test: {TEST_PROMPT}")
print(f"Model says: {adapted_response}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Before: {baseline_response}")
print(f"After:  {adapted_response}")
print(f"Expected: cold")