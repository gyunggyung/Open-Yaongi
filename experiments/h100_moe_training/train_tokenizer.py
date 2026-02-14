import os
import glob
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

DATA_DIR = "./data_cache"
OUTPUT_DIR = "./custom_tokenizer"
VOCAB_SIZE = 32768

def train_tokenizer():
    print("🚀 Starting Custom Tokenizer Training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Collect Data Files
    files = glob.glob(f"{DATA_DIR}/**/*.jsonl", recursive=True)
    if not files:
        print("❌ No data found in ./data_cache. Please run download_datasets.py first!")
        return
        
    print(f"📂 Found {len(files)} data files.")

    # 2. Define Special Tokens
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "<|im_start|>", "<|im_end|>", 
        "<|step|>", "<|code|>"
    ]
    
    # Add Reserved Tokens (for future use: RLHF, thoughts, tools)
    reserved_tokens = [f"<|reserved_{i}|>" for i in range(100)]
    all_special_tokens = special_tokens + reserved_tokens

    # 3. Initialize Tokenizer (BPE)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 4. Trainer Configuration
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=all_special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 5. Data Iterator
    def data_iterator():
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                # Sample 10% of lines to speed up training if files are huge
                # But for now, we assume download_datasets.py already saved a subset.
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        yield obj.get("text", "")
                    except:
                        pass

    # 6. Train
    print("⏳ Training Tokenizer (this may take a while)...")
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
    
    # 7. Post-Processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 8. Save
    print(f"💾 Saving to {OUTPUT_DIR}...")
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    
    # Save as HuggingFace compatible
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Tokenizer Training Complete!")
    
    # Validation
    test_texts = [
        "안녕하세요, 야옹이 모델입니다.",
        "def hello_world(): print('Hello')",
        "The quick brown fox jumps over the lazy dog."
    ]
    print("\n🧐 Validating Tokenizer:")
    for text in test_texts:
        encoded = fast_tokenizer.encode(text)
        decoded = fast_tokenizer.decode(encoded)
        print(f"Input: {text}")
        print(f"Tokens: {encoded}")
        print(f"Decoded: {decoded}")
        print("-" * 20)

if __name__ == "__main__":
    train_tokenizer()
