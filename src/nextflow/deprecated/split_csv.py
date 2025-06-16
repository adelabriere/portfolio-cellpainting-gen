#!/usr/bin/env python3
"""
Split a CSV file into multiple chunks while preserving the header in each chunk.

Usage:
    python split_csv.py input.csv chunk_size output_prefix
"""

import argparse
import csv
from typing import List


def split_csv(input_file: str, chunk_size: int, output_prefix: str) -> List[str]:
    """
    Split a CSV file into chunks while preserving the header in each file.

    Args:
        input_file: Path to the input CSV file
        chunk_size: Number of rows in each chunk (excluding header)
        output_prefix: Prefix for output files

    Returns:
        List of created output file paths
    """
    created_files = []
    
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header row
        
        chunk_num = 1
        current_chunk: List[List[str]] = []
        
        for row in reader:
            current_chunk.append(row)
            
            # If we've reached the chunk size, write the chunk to a file
            if len(current_chunk) >= chunk_size:
                output_file = f"{output_prefix}_{chunk_num}.csv"
                write_chunk(output_file, header, current_chunk)
                created_files.append(output_file)
                
                current_chunk = []  # Reset for the next chunk
                chunk_num += 1
        
        # Write any remaining rows
        if current_chunk:
            output_file = f"{output_prefix}_{chunk_num}.csv"
            write_chunk(output_file, header, current_chunk)
            created_files.append(output_file)
    
    return created_files


def write_chunk(output_file: str, header: List[str], chunk: List[List[str]]) -> None:
    """
    Write a chunk of data to a CSV file with the given header.

    Args:
        output_file: Path to the output file
        header: CSV header row
        chunk: List of data rows to write
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(chunk)


def main() -> None:
    """Parse command line arguments and split the CSV file."""
    parser = argparse.ArgumentParser(description='Split a CSV file into chunks with headers.')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('chunk_size', type=int, help='Number of rows in each chunk (excluding header)')
    parser.add_argument('output_prefix', help='Prefix for output files')
    
    args = parser.parse_args()
    
    created_files = split_csv(args.input_file, args.chunk_size, args.output_prefix)
    
    print(f"Split {args.input_file} into {len(created_files)} chunks:")
    for file in created_files:
        print(f"  - {file}")


if __name__ == "__main__":
    main()