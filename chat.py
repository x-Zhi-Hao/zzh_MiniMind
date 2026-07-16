"""Command-line chat for the final ZzhMind SFT checkpoint."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model import ZzhMindConfig, ZzhMindForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = PROJECT_ROOT / "weights" / "full_sft_real_v2_512.pth"


def load_model(weight_path: Path, device: str):
    """Load the tokenizer and the 25.8M, 512-dimension SFT model."""
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "model")
    model = ZzhMindForCausalLM(
        ZzhMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
    )
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device), tokenizer


@torch.inference_mode()
def reply(model, tokenizer, messages, device: str, max_new_tokens: int, temperature: float):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    do_sample = temperature > 0
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=0.9)
    generated_ids = model.generate(**generation_kwargs)
    new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with the ZzhMind SFT checkpoint")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Use cpu or cuda (default: auto-detect)",
    )
    parser.add_argument("--prompt", help="Ask one question and exit")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--history_turns", type=int, default=3)
    args = parser.parse_args()

    weight_path = args.weight.resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    model, tokenizer = load_model(weight_path, args.device)
    print(f"Loaded: {weight_path.name} on {args.device}")

    if args.prompt:
        answer = reply(
            model,
            tokenizer,
            [{"role": "user", "content": args.prompt}],
            args.device,
            args.max_new_tokens,
            args.temperature,
        )
        print(f"assistant: {answer}")
        return

    print("Type /exit to quit.")
    messages = []
    while True:
        question = input("you: ").strip()
        if question.lower() in {"/exit", "/quit"}:
            break
        if not question:
            continue

        messages.append({"role": "user", "content": question})
        answer = reply(
            model,
            tokenizer,
            messages,
            args.device,
            args.max_new_tokens,
            args.temperature,
        )
        print(f"assistant: {answer}")
        messages.append({"role": "assistant", "content": answer})
        if args.history_turns:
            messages = messages[-2 * args.history_turns :]


if __name__ == "__main__":
    main()
