import json
import os

def fix_notebook(path):
    print(f"Processing {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    
    # Strings to look for and replace
    replacements = {
        "LogisticRegression(max_iter=1000, random_state=42)": 
        "LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')",
        
        "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)": 
        "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')"
    }

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                modified_line = line
                for target, replacement in replacements.items():
                    if target in modified_line and "class_weight='balanced'" not in modified_line:
                        modified_line = modified_line.replace(target, replacement)
                        changed = True
                new_source.append(modified_line)
            cell['source'] = new_source

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {path} with class_weight='balanced'")
    else:
        print(f"No changes made to {path} (already updated or target not found)")

if __name__ == "__main__":
    fix_notebook('notebooks/05_modeling.ipynb')
    fix_notebook('notebooks/06_external_data_integration.ipynb')
