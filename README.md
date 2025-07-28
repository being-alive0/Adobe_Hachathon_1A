# README.md

# PDF Outline Extraction Challenge Submission

This project extracts a structured outline (title and headings) from PDF documents. It uses a sophisticated, modular, hybrid strategy that combines deterministic rule-based parsing with a machine learning model to achieve high accuracy across a wide variety of document layouts.

## Our Approach: A Modular, ToC-First Hybrid System

Our solution is built on a "ToC-First" principle, which treats the document's own Table of Contents (ToC) as the primary source of truth. The code is structured into two main modules for clarity and reusability:

- `outline_extractor.py`: A self-contained library with all the core logic for parsing and analyzing PDFs.
- `process_pdfs.py`: The main executable script that handles the Docker environment I/O and calls the extractor library.

The system follows two main paths:

**1. The "Golden Path": ToC-Based Extraction**
   - The script first intelligently scans the document to find a Table of Contents.
   - If a substantial ToC is found, it is parsed to extract the heading text, page numbers, and hierarchy (based on indentation).
   - This parsed ToC is used directly to generate the final JSON outline. This method is extremely reliable and acts as our primary, high-precision extraction engine.

**2. The "Inferential Path": ML-Based Fallback**
   - If no ToC is found, the system seamlessly switches to a fallback strategy powered by a pre-trained LightGBM machine learning model.
   - The ML model analyzes each line of the PDF based on a set of robust, layout-aware features.
   - **Features Used:**
     - `font_size`, `is_bold`, `font_family`
     - `is_centered`, `indentation`
     - `pattern_match` (for numeric prefixes like "1.1")
   - The model makes a binary prediction (`heading` / `not a heading`). The H1/H2/H3 levels are then assigned based on the relative font sizes of the predicted headings.

This hybrid architecture ensures the best of both worlds: the deterministic accuracy of a rule-based parser for documents that provide a ToC, and the flexible, inferential power of a machine learning model for documents that do not.

### Post-Processing
- **Multi-line Heading Stitching:** A post-processing step intelligently merges headings that span multiple lines.
- **NLP Sanity Check:** A final lightweight filter removes potential false positives that are clearly not human-readable headings (e.g., code snippets, gibberish).

## Models & Libraries Used

- **`PyMuPDF` (fitz):** For robust, high-performance PDF parsing.
- **`lightgbm`:** For the lightweight, fast, and accurate machine learning model used in the fallback strategy.
- **`pandas` & `numpy`:** For efficient data manipulation during feature engineering.

The pre-trained `submission_model_definitive.txt` is included. It was trained on a diverse set of sample documents to ensure it can generalize to various layouts.

## How to Build and Run

**Prerequisites:**
- Docker installed.

**Build the Docker Image:**
Place the `Dockerfile`, `process_pdfs.py`, `outline_extractor.py`, `requirements.txt`, and the pre-trained `submission_model_definitive.txt` in the same directory. Then, run the following command:
```sh
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .