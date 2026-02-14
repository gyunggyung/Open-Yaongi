import json

notebook_path = r"c:\github\Open-Yaongi\experiments\mamba2_ko_test\mamba2_colab_notebook.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify Config Cell (Index 2)
config_source = nb['cells'][2]['source']
new_config_source = []
for line in config_source:
    if "lr_muon =" in line:
        new_config_source.append('    lr_muon = 0.005     # Lowered significantly (0.02 -> 0.005)\n')
    elif "lr_adam =" in line:
        new_config_source.append('    lr_adam = 0.0005    # Lowered standard LR (0.001 -> 0.0005)\n')
    elif "batch_size =" in line:
        new_config_source.append('    batch_size = 64     # Reduced for safety\n')
    else:
        new_config_source.append(line)
nb['cells'][2]['source'] = new_config_source

# Modify Training Loop Cell (Index 5)
train_source = nb['cells'][5]['source']
new_train_source = []
for line in train_source:
    new_train_source.append(line)
    if "loss.backward()" in line:
        new_train_source.append('\n')
        new_train_source.append('        # Gradient Clipping (Essential for stability)\n')
        new_train_source.append('        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n')
        new_train_source.append('\n')

nb['cells'][5]['source'] = new_train_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("✅ Notebook patched successfully!")
