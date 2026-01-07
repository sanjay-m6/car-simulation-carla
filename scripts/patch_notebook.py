import json
import os

NOTEBOOK_PATH = "p:/Projects/Automanus-car/train_driving_model.ipynb"

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    found = False
    for cell in cells:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            # Identify training loop cell by unique content
            if any("for epoch in range(EPOCHS):" in line for line in source):
                
                print("Found training loop cell. Patching...")
                
                new_source = []
                # 1. Add import
                if not any("from evaluation.metrics import calculate_metrics" in line for line in source):
                    new_source.append("from evaluation.metrics import calculate_metrics\n")
                
                # Copy lines up to loop start
                loop_started = False
                for line in source:
                    new_source.append(line)
                    if "for epoch in range(EPOCHS):" in line:
                        loop_started = True
                    
                    # 2. Add list init inside validation block? No, inside loop before validation
                    # Actually better to init inside the epoch loop but outside validation loop
                    pass

                # Let's reconstruct more carefully
                new_source = []
                for line in source:
                    # Update import if it exists, or insert if missing
                    if "from evaluation.metrics import calculate_metrics" in line:
                         new_source.append("from evaluation.metrics import calculate_metrics, calculate_f1_score\n")
                    elif "for epoch in range(EPOCHS):" in line and not any("from evaluation.metrics" in s for s in new_source):
                         # Fallback if not found above (e.g. unpatched file)
                         new_source.insert(0, "from evaluation.metrics import calculate_metrics, calculate_f1_score\n")
                         new_source.append(line)
                    else:
                         new_source.append(line)
                    
                    # Init lists before validation loop
                    if "val_iter = iter(val_loader)" in line:
                        # Append after this line
                        new_source.append("\n")
                        new_source.append("    # Lists for metrics\n")
                        new_source.append("    val_outputs = []\n")
                        new_source.append("    val_targets = []\n")

                source = new_source
                new_source = []
                
                # 3. Collect data inside validation loop
                for line in source:
                    if "val_loss += criterion(model(images), controls).item()" in line:
                        # Need to capture output first
                        # Replace the line with multiple lines
                        # Maintain indentation
                        indent = line[:line.find("val_loss")]
                        new_source.append(f"{indent}output = model(images)\n")
                        new_source.append(f"{indent}val_loss += criterion(output, controls).item()\n")
                        new_source.append(f"{indent}val_outputs.append(output.cpu())\n")
                        new_source.append(f"{indent}val_targets.append(controls.cpu())\n")
                    else:
                        new_source.append(line)
                
                source = new_source
                new_source = []

                # 4. Calculate metrics at end of epoch (after validation loop)
                # Find unique anchor after validation loop
                for line in source:
                    if "avg_val = val_loss / max(val_steps, 1)" in line:
                        # Insert calculation block before this line
                        indent = "    "
                        new_source.append(f"{indent}if len(val_outputs) > 0:\n")
                        new_source.append(f"{indent}    all_preds = torch.cat(val_outputs, dim=0).numpy()\n")
                        new_source.append(f"{indent}    all_targs = torch.cat(val_targets, dim=0).numpy()\n")
                        new_source.append(f"{indent}    print(f\"\\n--- Epoch {{epoch+1}} Metrics ---\")\n")
                        new_source.append(f"{indent}    calculate_metrics(all_preds, all_targs)\n")
                        new_source.append(f"{indent}    calculate_f1_score(all_preds, all_targs)\n")
                        new_source.append("\n")
                    new_source.append(line)

                cell['source'] = new_source
                print("Patch applied to source.")
                found = True
                break
    
    if not found:
        print("Training loop cell not found!")
    else:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved.")

if __name__ == "__main__":
    patch_notebook()
