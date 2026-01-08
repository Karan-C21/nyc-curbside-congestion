"""
Apply class_weight='balanced' to models in Jupyter notebooks.

This utility script patches the modeling notebooks to use balanced class weights
for handling the imbalanced congestion prediction dataset.

Usage:
    python scripts/fix_class_imbalance.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NOTEBOOKS_DIR
from src.utils import setup_logging

# Setup logging
logger = setup_logging("notebook_patcher")

# Model instantiation patterns to update
REPLACEMENTS: Dict[str, str] = {
    "LogisticRegression(max_iter=1000, random_state=42)":
        "LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')",
    
    "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)":
        "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')"
}


def patch_notebook(notebook_path: Path) -> bool:
    """
    Patch a Jupyter notebook to use balanced class weights.
    
    Args:
        notebook_path: Path to the .ipynb file.
        
    Returns:
        True if changes were made, False otherwise.
    """
    logger.info(f"Processing {notebook_path.name}...")
    
    if not notebook_path.exists():
        logger.warning(f"Notebook not found: {notebook_path}")
        return False
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    changed = False
    
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        
        new_source: List[str] = []
        for line in cell.get("source", []):
            modified_line = line
            for target, replacement in REPLACEMENTS.items():
                if target in modified_line and "class_weight='balanced'" not in modified_line:
                    modified_line = modified_line.replace(target, replacement)
                    changed = True
            new_source.append(modified_line)
        cell["source"] = new_source
    
    if changed:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)
        logger.info(f"Updated {notebook_path.name} with class_weight='balanced'")
    else:
        logger.info(f"No changes needed for {notebook_path.name}")
    
    return changed


def main() -> None:
    """Main entry point for notebook patching."""
    notebooks_to_patch = [
        NOTEBOOKS_DIR / "05_modeling.ipynb",
        NOTEBOOKS_DIR / "06_external_data_integration.ipynb"
    ]
    
    total_changed = 0
    for notebook_path in notebooks_to_patch:
        if patch_notebook(notebook_path):
            total_changed += 1
    
    logger.info(f"Patched {total_changed}/{len(notebooks_to_patch)} notebooks")


if __name__ == "__main__":
    main()
