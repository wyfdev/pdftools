from sys import stderr
from pathlib import Path
from pymupdf import open as pdf_open


def merge_pdfs(input_files, output_file):
    """
    Merges PDF files from input_paths into output_path using pymupdf.

    Args:
        input_paths (list): A list of paths to the input PDF files.
        output_path (str): The path where the merged PDF should be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Create a new empty PDF Document object
    merged_doc = pdf_open()

    print("Starting mergeing")

    for pdf_path in input_files:
        if not Path(pdf_path).exists():
            print(f"Error:\n  Input file not found: {pdf_path}", file=stderr)
            merged_doc.close() # Close the potentially created doc before exiting
            return False
        if not pdf_path.lower().endswith('.pdf'):
             print(f"Warning: Input file does not have a .pdf extension: {pdf_path}", file=stderr)
             # Decide if you want to proceed or return False here
             # For now, we'll try to merge it anyway

        try:
            print(f"  >> adding: {pdf_path}")
            # Open the input document using pymupdf.open()
            with pdf_open(pdf_path) as input_doc:
                if input_doc.is_encrypted:
                     print(f"Error: Input file is encrypted and cannot be merged: {pdf_path}", file=stderr)
                     merged_doc.close()
                     return False
                # Insert the pages from input_doc into merged_doc
                # The method name remains the same on the Document object
                merged_doc.insert_pdf(input_doc)
        # Catch exceptions specific to pymupdf (FitzError is often available directly)
        except Exception as e:
            print(f"An unexpected error occurred with file {pdf_path}: {e}", file=stderr)
            merged_doc.close()
            return False

    try:
        # Save the merged document
        # Method name and options remain the same on the Document object
        merged_doc.save(output_file, garbage=4, deflate=True)
        print(f'Successfully merge {len(input_files)} files')
        print(f'    saveing: "{output_file}"')
        return True
    except Exception as e:
        print(f"Error saving the merged file {output_file}: {e}", file=stderr)
        return False
    finally:
        merged_doc.close()
