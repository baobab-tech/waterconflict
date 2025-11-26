#!/usr/bin/env python3
"""
Analyze unique values (enums) for categorical columns in ACLED data
"""
import csv
from collections import defaultdict
import json

def analyze_enums(csv_file, output_file='data/enum_analysis.json'):
    """Extract unique values for categorical columns"""
    
    # Columns to analyze for unique values
    enum_columns = [
        'disorder_type',
        'event_type', 
        'sub_event_type',
        'inter1',
        'inter2',
        'interaction',
        'civilian_targeting',
        'region',
        'source_scale'
    ]
    
    # Dictionary to store unique values
    unique_values = {col: set() for col in enum_columns}
    total_rows = 0
    
    print(f"Analyzing {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            for col in enum_columns:
                value = row.get(col, '').strip()
                if value:  # Only add non-empty values
                    unique_values[col].add(value)
            
            # Progress indicator
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print("\nUnique values found:")
    
    # Convert sets to sorted lists and print summary
    results = {}
    for col in enum_columns:
        sorted_values = sorted(list(unique_values[col]))
        results[col] = sorted_values
        print(f"  {col}: {len(sorted_values)} unique values")
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print detailed breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    
    for col in enum_columns:
        print(f"\n{col.upper()} ({len(results[col])} values):")
        print("-" * 40)
        for value in results[col]:
            print(f"  - {value}")
    
    return results

if __name__ == '__main__':
    csv_file = 'data/ACLED RAW_2024-01-01-2025-05-19.csv'
    analyze_enums(csv_file)

