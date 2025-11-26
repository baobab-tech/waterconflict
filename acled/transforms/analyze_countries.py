#!/usr/bin/env python3
"""
Analyze country and region distributions in ACLED data
"""
import csv
from collections import Counter
import json

def analyze_countries(csv_file, output_file='data/country_analysis.json'):
    """Extract country, region, and geographic distribution"""
    
    countries = Counter()
    regions = Counter()
    country_to_region = {}
    country_iso = {}
    
    total_rows = 0
    
    print(f"Analyzing geographic distribution in {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            country = row.get('country', '').strip()
            region = row.get('region', '').strip()
            iso = row.get('iso', '').strip()
            
            if country:
                countries[country] += 1
                if region:
                    country_to_region[country] = region
                if iso:
                    country_iso[country] = iso
            
            if region:
                regions[region] += 1
            
            if total_rows % 50000 == 0:
                print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print(f"Total countries: {len(countries)}")
    print(f"Total regions: {len(regions)}")
    
    # Prepare results
    results = {
        'total_events': total_rows,
        'total_countries': len(countries),
        'total_regions': len(regions),
        'regions': [
            {'name': region, 'events': count}
            for region, count in regions.most_common()
        ],
        'countries': [
            {
                'name': country,
                'events': count,
                'region': country_to_region.get(country, ''),
                'iso_code': country_iso.get(country, '')
            }
            for country, count in countries.most_common()
        ]
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("REGIONS")
    print("="*80)
    for region_data in results['regions']:
        print(f"  {region_data['name']}: {region_data['events']:,} events")
    
    print("\n" + "="*80)
    print("TOP 30 COUNTRIES")
    print("="*80)
    for country_data in results['countries'][:30]:
        print(f"  {country_data['name']:30s} | {country_data['region']:20s} | {country_data['events']:,} events")
    
    return results

if __name__ == '__main__':
    csv_file = 'data/ACLED RAW_2024-01-01-2025-05-19.csv'
    analyze_countries(csv_file)

