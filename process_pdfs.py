# process_pdfs.py

import os
import json
from pathlib import Path
import lightgbm as lgb

# Import the core processing function from our library module
from outline_extractor import process_document_definitive

def process_pdfs():
    """
    Main function to run inside the Docker container.
    """
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    model_path = Path("/app/submission_model_definitive.txt")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the pre-trained model
    model = None
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        print("Pre-trained model loaded successfully.")
    else:
        print(f"WARNING: Model file not found at {model_path}. The script will only be able to process PDFs with a Table of Contents.")

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in /app/input.")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()
            
            # Call the core logic from our library
            json_output_str = process_document_definitive(pdf_bytes, model)
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            with open(output_file, "w") as f:
                f.write(json_output_str)
            
            print(f"Processed {pdf_file.name} -> {output_file.name}")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            error_output = {
                "title": f"Error processing {pdf_file.name}",
                "outline": [{"level": "ERROR", "text": str(e), "page": 0}]
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(error_output, f, indent=2)

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("completed processing pdfs")