import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from app.config import Settings


class ModelService:
    """
    Wraps model loading and inference.

    To swap models after retraining, update Settings.base_model_name and
    Settings.adapter_path — no changes needed here.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        s = self.settings

        # Device + dtype selection
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32

        tokenizer = AutoTokenizer.from_pretrained(s.adapter_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            s.base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        base_model.resize_token_embeddings(len(tokenizer))

        # Merge LoRA adapter — eliminates float16/float32 mismatch on CPU/MPS
        model = PeftModel.from_pretrained(base_model, s.adapter_path).merge_and_unload()
        model = model.to(dtype=dtype, device=device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

        print(f"[ModelService] {s.base_model_name} + {s.adapter_path} | device={device} | version={s.model_version}")

    def complete(self, prefix: str, suffix: str, max_new_tokens: int) -> str:
        s = self.settings
        prompt = f"{s.fim_prefix}{prefix}{s.fim_suffix}{suffix}{s.fim_middle}"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=s.max_input_length,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        if generated.startswith(prompt_decoded):
            return generated[len(prompt_decoded):].strip()
        return generated.strip()

    @property
    def version(self) -> str:
        return self.settings.model_version
