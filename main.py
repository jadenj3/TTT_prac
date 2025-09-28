import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = "Qwen/Qwen2.5-7B-Instruct"        # or "Qwen/Qwen1.5-0.5B"
DTYPE = torch.float16           # bfloat16 also works on A100/H100

tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    torch_dtype=DTYPE,
    device_map={"": "cuda"},    # put everything on GPU-0
    trust_remote_code=True,
)

print(f"Loaded {model.num_parameters()/1e6:.0f} M params")
device  = torch.device("cuda")