#!/usr/bin/env python3
"""
Analyze tags and temporal patterns in ACLED data
"""
import csv
from collections import Counter, defaultdict
import json
from datetime import datetime

def analyze_tags_and_temporal(csv_file, output_file='data/tags_temporal_analysis.json'):
    """Analyze tags and temporal patterns"""
    
    tags_counter = Counter()
    years_counter = Counter()
    months_counter = Counter()
    precision_counter = Counter()
    geo_precision_counter = Counter()
    
    fatalities_distribution = Counter()
    events_with_fatalities = 0
    total_fatalities = 0
    
    total_rows = 0
    
    print(f"Analyzing tags and temporal patterns in {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            # Tags
            tags = row.get('tags', '').strip()
            if tags:
                tags_counter[tags] += 1
            
            # Year
            year = row.get('year', '').strip()
            if year:
                years_counter[year] += 1
            
            # Month from event_date
            event_date = row.get('event_date', '').strip()
            if event_date:
                try:
                    # Parse date like "16 May 2025"
                    date_obj = datetime.strptime(event_date, '%d %B %Y')
                    month_key = date_obj.strftime('%Y-%m')
                    months_counter[month_key] += 1
                except:
                    pass
            
            # Time precision
            time_prec = row.get('time_precision', '').strip()
            if time_prec:
                precision_counter[time_prec] += 1
            
            # Geo precision
            geo_prec = row.get('geo_precision', '').strip()
            if geo_prec:
                geo_precision_counter[geo_prec] += 1
            
            # Fatalities
            fatalities = row.get('fatalities', '').strip()
            if fatalities and fatalities.isdigit():
                fat_count = int(fatalities)
                if fat_count > 0:
                    events_with_fatalities += 1
                    total_fatalities += fat_count
                
                # Bin fatalities
                if fat_count == 0:
                    fatalities_distribution['0'] += 1
                elif fat_count <= 5:
                    fatalities_distribution['1-5'] += 1
                elif fat_count <= 10:
                    fatalities_distribution['6-10'] += 1
                elif fat_count <= 50:
                    fatalities_distribution['11-50'] += 1
                elif fat_count <= 100:
                    fatalities_distribution['51-100'] += 1
                else:
                    fatalities_distribution['100+'] += 1
            
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print(f"Unique tags: {len(tags_counter):,}")
    print(f"Events with fatalities: {events_with_fatalities:,}")
    print(f"Total fatalities: {total_fatalities:,}")
    
    # Prepare results
    results = {
        'total_events': total_rows,
        'events_with_fatalities': events_with_fatalities,
        'total_fatalities': total_fatalities,
        'average_fatalities_per_deadly_event': round(total_fatalities / events_with_fatalities, 2) if events_with_fatalities > 0 else 0,
        'years': [
            {'year': year, 'events': count}
            for year, count in sorted(years_counter.items())
        ],
        'months': [
            {'month': month, 'events': count}
            for month, count in sorted(months_counter.items())
        ],
        'time_precision': [
            {'precision': prec, 'count': count}
            for prec, count in precision_counter.most_common()
        ],
        'geo_precision': [
            {'precision': prec, 'count': count}
            for prec, count in geo_precision_counter.most_common()
        ],
        'fatalities_distribution': [
            {'range': range_name, 'count': count}
            for range_name, count in sorted(fatalities_distribution.items(), key=lambda x: (
                0 if x[0] == '0' else
                1 if x[0] == '1-5' else
                2 if x[0] == '6-10' else
                3 if x[0] == '11-50' else
                4 if x[0] == '51-100' else 5
            ))
        ],
        'top_50_tags': [
            {'tag': tag, 'count': count}
            for tag, count in tags_counter.most_common(50)
        ]
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summaries
    print("\n" + "="*80)
    print("TEMPORAL DISTRIBUTION")
    print("="*80)
    print("\nYears:")
    for item in results['years']:
        print(f"  {item['year']}: {item['events']:,} events")
    
    print("\n" + "="*80)
    print("FATALITIES DISTRIBUTION")
    print("="*80)
    for item in results['fatalities_distribution']:
        print(f"  {item['range']:10s} | {item['count']:,} events")
    
    print("\n" + "="*80)
    print("TOP 20 TAGS")
    print("="*80)
    for item in results['top_50_tags'][:20]:
        print(f"  {item['tag']:50s} | {item['count']:,} events")
    
    print("\n" + "="*80)
    print("PRECISION LEVELS")
    print("="*80)
    print("\nTime Precision:")
    for item in results['time_precision']:
        print(f"  Level {item['precision']}: {item['count']:,} events")
    print("\nGeo Precision:")
    for item in results['geo_precision']:
        print(f"  Level {item['precision']}: {item['count']:,} events")
    
    return results

if __name__ == '__main__':
    csv_file = 'data/ACLED RAW_2024-01-01-2025-05-19.csv'
    analyze_tags_and_temporal(csv_file)

