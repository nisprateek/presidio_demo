#!/usr/bin/env python
"""
Command-line tool to anonymize text files using Microsoft Presidio.
Processes all files in an input folder and saves the anonymized versions to an output folder.
"""

import os
import argparse
import logging
from pathlib import Path
import dotenv
import json
from typing import Dict, Optional

from src.presidio_helpers import (
    analyze,
    anonymize,
    get_supported_entities,
    analyzer_engine,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("presidio-cli")

# Set up file handler for error logging
error_log_handler = logging.FileHandler("anonymization_errors.log")
error_log_handler.setLevel(logging.ERROR)
error_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_log_handler.setFormatter(error_log_formatter)
logger.addHandler(error_log_handler)

# Load environment variables
dotenv.load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Anonymize text files using Microsoft Presidio")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Path to folder containing files to anonymize")
    parser.add_argument("--output_folder", type=str, required=True, 
                        help="Path to folder where anonymized files will be saved")
    parser.add_argument("--model", type=str, default="flair/ner-english-large",
                        choices=["flair/ner-english-large", "OpenAI/GPT_4o"],
                        help="NER model to use for detection")
    parser.add_argument("--anonymization", type=str, default="replace",
                        choices=["redact", "replace", "highlight", "mask", "hash", "encrypt"],
                        help="Anonymization technique to apply")
    parser.add_argument("--entities", type=str, nargs="*", 
                        help="List of entities to detect (default: all)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Confidence threshold for accepting a detection as PII")
    parser.add_argument("--mask_char", type=str, default="*",
                        help="Character to use for masking (if using mask)")
    parser.add_argument("--num_chars", type=int, default=15,
                        help="Number of characters to mask (if using mask)")
    parser.add_argument("--encrypt_key", type=str, default="WmZq4t7w!z%C&F)J",
                        help="Key for encryption (if using encrypt)")
    parser.add_argument("--custom_entities", type=str, default=None,
                        help="Path to JSON file defining custom entities")
    
    return parser.parse_args()

def is_binary_file(file_path):
    """Check if a file is binary (non-text)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Try to read first 1024 bytes
        return False
    except UnicodeDecodeError:
        # Try with latin-1 which can read any byte value
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                chunk = f.read(1024)
                # If high concentration of nulls or non-printable chars, likely binary
                null_count = chunk.count('\x00')
                control_count = sum(1 for c in chunk if ord(c) < 32 and c not in '\n\r\t')
                return null_count > 10 or (control_count / len(chunk) > 0.3 if chunk else 0)
        except:
            return True
    except:
        return True

def process_file(file_path, output_path, model_package, model_name, anonymization, entities, 
                 threshold, mask_char=None, num_chars=None, encrypt_key=None, custom_entities=None):
    """Process a single file."""
    logger.info(f"Processing file: {file_path}")
    
    # Check if file is likely binary
    if is_binary_file(file_path):
        error_msg = f"Skipping binary file: {file_path}"
        logger.error(error_msg)
        return 0
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            logger.warning(f"Unicode decode error with {file_path}, trying with latin-1 encoding")
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {e}"
            logger.error(error_msg)
            return 0
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {e}"
        logger.error(error_msg)
        return 0
    
    # Analyze the text
    try:
        results = analyze(
            model_package, model_name,
            text=text,
            entities=entities,
            language="en",
            score_threshold=threshold,
            custom_entities=custom_entities,
        )
    except Exception as e:
        error_msg = f"Error analyzing file {file_path}: {e}"
        logger.error(error_msg)
        return 0
    
    # Anonymize the text
    try:
        anonymized_results = anonymize(
            text=text,
            operator=anonymization,
            mask_char=mask_char,
            number_of_chars=num_chars,
            encrypt_key=encrypt_key,
            analyze_results=results,
        )
    except Exception as e:
        error_msg = f"Error anonymizing file {file_path}: {e}"
        logger.error(error_msg)
        return 0
    
    # Write the anonymized text to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_results.text)
        logger.info(f"Saved anonymized file to: {output_path}")
        return len(results)
    except Exception as e:
        error_msg = f"Error writing anonymized file {output_path}: {e}"
        logger.error(error_msg)
        return 0

def load_custom_entities(json_path: str) -> Optional[Dict[str, Dict[str, str]]]:
    """Load custom entities from a JSON file."""
    if not json_path:
        return None
        
    try:
        with open(json_path, 'r') as f:
            custom_entities = json.load(f)
        
        # Validate format
        for entity_name, entity_info in custom_entities.items():
            if not isinstance(entity_info, dict):
                logger.error(f"Invalid format for entity {entity_name}. Expected dict, got {type(entity_info)}")
                return None
                
            if 'description' not in entity_info:
                logger.warning(f"No description provided for custom entity {entity_name}")
                custom_entities[entity_name]['description'] = f"Custom entity type: {entity_name}"
                
        return custom_entities
    except Exception as e:
        logger.error(f"Error loading custom entities file: {e}")
        return None

def main():
    """Main entry point."""
    args = parse_args()
    
    # Extract model package and name
    model_package, model_name = args.model.lower().split("/")
    
    # Load custom entities if specified
    custom_entities = load_custom_entities(args.custom_entities)
    if custom_entities:
        logger.info(f"Loaded {len(custom_entities)} custom entity types")
    
    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get entities to detect
    if args.entities:
        entities = args.entities
        # Add custom entities to the detection list
        if custom_entities:
            entities = list(set(entities + list(custom_entities.keys())))
    else:
        entities = get_supported_entities(model_package, model_name)
        logger.info(f"Using all supported entities: {entities}")
    
    # Process all files in the input folder
    input_folder = Path(args.input_folder)
    
    # Check if the input folder exists
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return
    
    # Initialize counters
    file_count = 0
    processed_count = 0
    entity_count = 0
    error_count = 0
    
    # Initialize the analyzer engine with custom entities
    logger.info(f"Initializing analyzer with model: {args.model}")
    analyzer = analyzer_engine(model_package, model_name, custom_entities=custom_entities)
    
    # List all files in the directory
    logger.info(f"Scanning input folder: {input_folder}")
    
    # Collect all files (not directories)
    files_to_process = []
    for item in input_folder.glob("**/*"):
        if item.is_file():
            files_to_process.append(item)
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for file_path in files_to_process:
        file_count += 1
        
        relative_path = file_path.relative_to(input_folder)
        output_path = output_folder / relative_path
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the file
        try:
            entities_found = process_file(
                file_path, 
                output_path, 
                model_package, 
                model_name, 
                args.anonymization,
                entities, 
                args.threshold,
                args.mask_char,
                args.num_chars,
                args.encrypt_key,
                custom_entities=custom_entities
            )
            
            if entities_found is not None:
                processed_count += 1
                entity_count += entities_found
            else:
                error_count += 1
        except Exception as e:
            error_msg = f"Uncaught exception processing {file_path}: {str(e)}"
            logger.error(error_msg)
            error_count += 1
    
    logger.info(f"Processing complete! Found {file_count} files")
    logger.info(f"Successfully processed {processed_count} files")
    logger.info(f"Failed to process {error_count} files (see anonymization_errors.log)")
    logger.info(f"Found {entity_count} entities across all processed files")

if __name__ == "__main__":
    main() 