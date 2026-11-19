import os
import sys
import argparse
from typing import Dict, Any

def create_mock_preference_dataset() -> Any:
    """
    Creates a mock DPO (Direct Preference Optimization) dataset of academic tutor
    conversations. Maps prompts to 'chosen' (grounded + correctly cited responses)
    and 'rejected' (hallucinated or uncited responses) to align the SLM.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("Error: 'datasets' library is missing. Install using: pip install datasets")
        sys.exit(1)
        
    data = {
        "prompt": [
            "Explain Unit 3 on Decision Trees.",
            "What is the standard value of parameter b in BM25?",
            "How does semantic re-ranking improve RAG answers?",
            "What chunking strategy should be used for lecture slides?"
        ],
        "chosen": [
            "Decision Trees are hierarchical models used for classification and regression. They split data on features that maximize information gain [1]. Each split creates child nodes representing decisions, ending at terminal leaf nodes containing predicted outputs [2].",
            "In Okapi BM25 keyword search, the parameter b controls document length normalization. The standard default value used in RAG search systems is b = 0.75 [1]. This penalizes longer documents to prevent dense, unrelated text from ranking highly [2].",
            "A Cross-Encoder semantic re-ranker evaluates query-document pairs jointly to calculate a deep contextual relevance score [1]. By re-ordering candidates, it successfully promotes the absolute most relevant chunks to the top of the context block [2].",
            "Lecture slides are highly thematic and discrete. Therefore, a slide-based chunking strategy is recommended [1]. This strategy processes each slide as an individual chunk, keeping slide titles, bullet points, tables, and speaker notes grouped together [2]."
        ],
        "rejected": [
            "Decision trees are models that split data on features to make decisions. They are very fast and can be trained in any environment without libraries. They do not have any constraints on information gain.",
            "Okapi BM25 has multiple parameters like b and k1. b is normally set to 1.0 or 0.0 depending on the length of your text, but it doesn't matter much in simple databases.",
            "Re-ranking works by sorting documents using their cosine similarities. It runs on the CPU and resolves all similarity queries in under one millisecond using SQLite, without needing deep learning models.",
            "For slides, you should treat the entire PowerPoint file as a single large document. Simply dump all extracted characters into one large chunk of 5000 characters without separating slides."
        ]
    }
    
    dataset = Dataset.from_dict(data)
    print(f"[PREFERENCE BUILDER] Scaffolding local DPO preference dataset with {len(dataset)} examples...")
    return dataset

def run_finetuning(args: argparse.Namespace):
    """
    Loads base models, binds LoRA adapters, sets up DPOTrainer,
    and runs local SLM preference alignment.
    """
    print("=" * 60)
    print("🎓 EduQuery AI - Local SLM LoRA + DPO Fine-Tuning Pipeline")
    print("=" * 60)
    
    # 1. CUDA Telemetry & Safety Checks
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Please install PyTorch to run local training.")
        return
        
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"[SYSTEM CHECK] PyTorch Active Device: {device.upper()}")
    
    if not cuda_available:
        print("\n" + "!" * 60)
        print("⚠️ WARNING: NVIDIA CUDA GPU was NOT detected!")
        print("Local LLM training on a CPU is extremely slow and likely to run out of RAM.")
        print("It is highly recommended to run this training pipeline on a machine with a")
        print("dedicated NVIDIA GPU containing at least 12GB of VRAM (RTX 3060/4060 or higher).")
        print("!" * 60 + "\n")
        if not args.force_cpu:
            print("Aborting training run. Use --force_cpu to override and run on CPU.")
            return
            
    # 2. Lazy load training modules
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import DPOTrainer
    except ImportError as e:
        print(f"Error: Missing advanced deep learning libraries: {e}")
        print("Install required training packages: pip install transformers peft trl accelerate")
        return
        
    # 3. Load DPO Preference dataset
    dpo_dataset = create_mock_preference_dataset()
    
    print(f"\n[MODEL LOADING] Initializing base model: '{args.model_id}'...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configure standard LoRA Low-Rank Adaptation parameters
        print(f"[LORA CONFIG] Scaffolding PEFT parameters: Rank={args.lora_r}, Alpha={args.lora_alpha}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Standard attention projection targets
        )
        
        # Load Model
        # When running locally on GPU, use 8-bit or 4-bit quantization to prevent VRAM out-of-memory errors
        model_kwargs = {}
        if cuda_available:
            model_kwargs = {
                "device_map": "auto",
                "load_in_8bit": True if args.load_8bit else False
            }
            
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_kwargs
        )
        
        print("[PEFT SETUP] Binding LoRA trainable adapter weights to base model...")
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
        
        # Standard DPO Training Arguments
        print(f"\n[TRAINING ARGS] Setting up hyper-parameters (Learning Rate: {args.learning_rate})...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            remove_unused_columns=False,
            report_to="none"
        )
        
        # 4. DPOTrainer initialization
        print("[DPO TRAINER] Initializing Direct Preference Optimization Trainer...")
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            beta=0.1,  # Implicit reward scale parameter for DPO
            train_dataset=dpo_dataset,
            tokenizer=tokenizer,
            max_length=512,
            max_prompt_length=256
        )
        
        print("\n" + "=" * 60)
        print("🚀 STARTING LOCAL SLM FINE-TUNING...")
        print("=" * 60)
        
        # Start training
        dpo_trainer.train()
        
        # Save custom LoRA PEFT adapter weights
        adapter_output_dir = os.path.join(args.output_dir, "lora_adapters")
        print(f"\n[SAVING] Fine-tuning completed. Saving custom LoRA adapters to: {adapter_output_dir}")
        model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        print("Success! You can now load these custom adapters in Ollama or your backend server.")
        
    except Exception as err:
        print(f"\n❌ Fine-tuning execution failed: {err}")
        print("Please check your PyTorch installation, CUDA path, or Hugging Face token rights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune local SLMs for strict grounded citations.")
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Hugging Face base model ID to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./backend/rag/finetune_output", help="Directory to save LoRA PEFT adapters")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of local training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Adam optimizer learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA matrix rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scale factor alpha")
    parser.add_argument("--load_8bit", action="store_true", help="Load base model in 8-bit quantization for VRAM efficiency")
    parser.add_argument("--force_cpu", action="store_true", help="Override CUDA safety checks and execute training on CPU")
    
    args = parser.parse_args()
    run_finetuning(args)
