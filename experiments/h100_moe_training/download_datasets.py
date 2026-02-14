import os
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Configuration from the Plan
DATASET_CONFIG = {
    # 1. Korean (40%)
    "korean": [
        {"path": "HAERAE-HUB/KOREAN-SyntheticText-1.5B", "ratio": 0.20},
        {"path": "KORMo-Team/Cosmopedia-ko-synth",      "ratio": 0.10},
        {"path": "lcw99/wikipedia-korean-20240501",       "ratio": 0.05},
        {"path": "KORMo-Team/FineWeb2-ko-synth",         "ratio": 0.05},
    ],
    # 2. English (20%)
    "english": [
        {"path": "allenai/dolma3_dolmino_mix-10B-1025",  "ratio": 0.10},
        {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "ratio": 0.10},
    ],
    # 3. Code (25%)
    "code": [
        {"path": "allenai/Dolci-Think-SFT-Python",       "ratio": 0.10},
        {"path": "HuggingFaceTB/stack-edu", "name": "Python", "ratio": 0.15},
    ],
    # 4. Math (15%)
    "math": [
        {"path": "HuggingFaceTB/finemath", "name": "finemath-4plus", "ratio": 0.15},
    ]
}

OUTPUT_DIR = "./data_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_and_save(config):
    """
    Downloads datasets and saves them to disk in a simple format (e.g., JSONL or Parquet)
    for fast loading during training. We will use `load_dataset` with `streaming=True`
    to avoid disk explosion during download, but save a subset/shard locally.
    """
    print(f"🚀 Starting Dataset Download to {OUTPUT_DIR}...")
    
    summary = {}
    
    for category, datasets in config.items():
        print(f"\n📂 Processing Category: {category}")
        category_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for ds_info in datasets:
            path = ds_info['path']
            name = ds_info.get('name', None)
            ratio = ds_info['ratio']
            
            safe_name = path.replace("/", "_")
            if name: safe_name += f"_{name}"
            output_path = os.path.join(category_dir, f"{safe_name}.jsonl")
            
            if os.path.exists(output_path):
                print(f"  ✅ Skipped {path} (Already exists)")
                continue
                
            print(f"  ⬇️ Downloading {path} (ratio: {ratio})...")
            
            try:
                # Load streaming dataset
                ds = load_dataset(path, name, split="train", streaming=True)
                
                # Determine text field
                text_field = 'text'
                # Special cases for code/chat datasets
                first_ex = next(iter(ds))
                if 'messages' in first_ex: text_field = 'messages'
                elif 'content' in first_ex: text_field = 'content'
                elif 'instruction' in first_ex and 'output' in first_ex: text_field = ['instruction', 'output']
                elif 'prompt' in first_ex and 'solution' in first_ex: text_field = ['prompt', 'solution']
                
                # Save a reasonable amount of data (e.g., 100k samples for PoC, or full for real)
                # For this script, we'll save up to 100,000 samples per dataset as a "Check"
                # In real training on H100 with fast internet, you might stream directly or save millions.
                LIMIT = 50000 
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    count = 0
                    for i, example in enumerate(tqdm(ds)):
                        if i >= LIMIT: break
                        
                        # Extract content
                        content = ""
                        if isinstance(text_field, list):
                            for field in text_field:
                                content += f"{field.upper()}: {example.get(field, '')}\n"
                        elif text_field == 'messages':
                            # Simplify chat to text
                            for msg in example.get('messages', []):
                                content += f"{msg.get('role', '')}: {msg.get('content', '')}\n"
                        else:
                            content = example.get(text_field, "")
                            
                        if not content: continue
                        
                        # Write as JSONL
                        json.dump({"text": content}, f, ensure_ascii=False)
                        f.write('\n')
                        count += 1
                        
                summary[path] = count
                print(f"  ✅ Saved {count} samples to {output_path}")
                
            except Exception as e:
                print(f"  ❌ Failed to download {path}: {e}")
    
    print("\n🎉 Download Complete!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    download_and_save(DATASET_CONFIG)
