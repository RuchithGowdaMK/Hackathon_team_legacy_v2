from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GraniteClient:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu':
            print("WARNING: CUDA not available. Running on CPU (slow).")
        else:
            print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

        model_name = "ibm-granite/granite-3.2-2b-instruct"
        print(f"Loading {model_name}...")

        # Use trust_remote_code=True to support custom Granite architecture
        print("[1] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,            # Slow tokenizer for maximum compatibility
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[1] Tokenizer loaded successfully")

        print("[2] Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,   # CRITICAL for custom model types
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            low_cpu_mem_usage=True
        )

        if self.device == 'cpu':
            self.model = self.model.to('cpu')

        print("[2] Model loaded successfully.")
        print("Granite model ready for inference!")

    def generate_answer(self, question: str, context: str, max_new_tokens=100) -> str:
        print(f"\n{'='*80}")
        print(f"[GRANITE] Starting answer generation...")
        print(f"[GRANITE] Device: {self.device}")
        print(f"[GRANITE] Question: {question[:60]}...")
        print(f"{'='*80}")

        prompt = f"""You are an expert academic assistant. Answer ONLY based on the context provided.
If the answer is not in the context, respond: "Information not found in the document."

Context:
{context}

Question: {question}

Answer:"""

        print("[GRANITE] Tokenizing input...")

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )

            if self.device == 'cuda':
                print("[GRANITE] Moving inputs to CUDA...")
                inputs = {k: v.cuda() for k, v in inputs.items()}

            print("[GRANITE] Generating answer (this may take 3-8 seconds)...")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    num_beams=1
                )

            print("[GRANITE] Generation complete.")

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response.strip()

            print(f"[GRANITE] Answer: {answer[:150]}...")
            print(f"{'='*80}\n")

            return answer

        except Exception as e:
            print(f"[GRANITE ERROR] {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
