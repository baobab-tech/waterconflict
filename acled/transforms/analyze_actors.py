#!/usr/bin/env python3
"""
Analyze actor patterns in ACLED data
"""
import csv
from collections import Counter
import json
import re

def extract_actor_base(actor_string):
    """Extract base actor type from full actor string"""
    if not actor_string:
        return None
    
    # Extract text before country/year info in parentheses
    match = re.match(r'^([^(]+)', actor_string)
    if match:
        return match.group(1).strip()
    return actor_string.strip()

def analyze_actors(csv_file, output_file='data/actor_analysis.json'):
    """Analyze actor types and patterns"""
    
    actor1_counter = Counter()
    actor2_counter = Counter()
    actor_base_types = Counter()
    interaction_types = Counter()
    
    total_rows = 0
    
    print(f"Analyzing actors in {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            actor1 = row.get('actor1', '').strip()
            actor2 = row.get('actor2', '').strip()
            interaction = row.get('interaction', '').strip()
            
            if actor1:
                actor1_counter[actor1] += 1
                base = extract_actor_base(actor1)
                if base:
                    actor_base_types[base] += 1
            
            if actor2:
                actor2_counter[actor2] += 1
                base = extract_actor_base(actor2)
                if base:
                    actor_base_types[base] += 1
            
            if interaction:
                interaction_types[interaction] += 1
            
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print(f"Unique Actor1 values: {len(actor1_counter):,}")
    print(f"Unique Actor2 values: {len(actor2_counter):,}")
    print(f"Base actor types: {len(actor_base_types):,}")
    print(f"Interaction types: {len(interaction_types):,}")
    
    # Prepare results
    results = {
        'total_events': total_rows,
        'unique_actor1_count': len(actor1_counter),
        'unique_actor2_count': len(actor2_counter),
        'interaction_types': [
            {'type': interaction, 'count': count}
            for interaction, count in interaction_types.most_common()
        ],
        'top_100_actor1': [
            {'actor': actor, 'count': count}
            for actor, count in actor1_counter.most_common(100)
        ],
        'top_100_actor2': [
            {'actor': actor, 'count': count}
            for actor, count in actor2_counter.most_common(100)
        ],
        'top_50_base_actor_types': [
            {'type': actor_type, 'count': count}
            for actor_type, count in actor_base_types.most_common(50)
        ]
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summaries
    print("\n" + "="*80)
    print("INTERACTION TYPES")
    print("="*80)
    for item in results['interaction_types']:
        print(f"  {item['type']:50s} | {item['count']:,} events")
    
    print("\n" + "="*80)
    print("TOP 20 BASE ACTOR TYPES")
    print("="*80)
    for item in results['top_50_base_actor_types'][:20]:
        print(f"  {item['type']:50s} | {item['count']:,} appearances")
    
    print("\n" + "="*80)
    print("TOP 20 ACTOR1 (Primary Actors)")
    print("="*80)
    for item in results['top_100_actor1'][:20]:
        print(f"  {item['actor']:60s} | {item['count']:,} events")
    
    return results

if __name__ == '__main__':
    csv_file = 'data/ACLED RAW_2024-01-01-2025-05-19.csv'
    analyze_actors(csv_file)

