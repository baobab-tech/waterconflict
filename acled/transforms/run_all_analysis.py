#!/usr/bin/env python3
"""
Master analysis script - runs all ACLED data analyses and generates comprehensive output
"""
import json
import time
from datetime import datetime
from analyze_enums import analyze_enums
from analyze_countries import analyze_countries
from analyze_actors import analyze_actors
from analyze_tags_and_temporal import analyze_tags_and_temporal

def run_all_analyses(csv_file='data/ACLED RAW_2024-01-01-2025-05-19.csv'):
    """Run all analysis scripts and combine results"""
    
    print("="*80)
    print("ACLED DATA COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Run each analysis
    results = {}
    
    print("\n" + "="*80)
    print("1/4 - ANALYZING ENUM VALUES...")
    print("="*80)
    enum_start = time.time()
    results['enums'] = analyze_enums(csv_file, 'data/enum_analysis.json')
    print(f"Completed in {time.time() - enum_start:.1f} seconds")
    
    print("\n" + "="*80)
    print("2/4 - ANALYZING GEOGRAPHIC DISTRIBUTION...")
    print("="*80)
    geo_start = time.time()
    results['geography'] = analyze_countries(csv_file, 'data/country_analysis.json')
    print(f"Completed in {time.time() - geo_start:.1f} seconds")
    
    print("\n" + "="*80)
    print("3/4 - ANALYZING ACTORS...")
    print("="*80)
    actor_start = time.time()
    results['actors'] = analyze_actors(csv_file, 'data/actor_analysis.json')
    print(f"Completed in {time.time() - actor_start:.1f} seconds")
    
    print("\n" + "="*80)
    print("4/4 - ANALYZING TAGS AND TEMPORAL PATTERNS...")
    print("="*80)
    tags_start = time.time()
    results['tags_temporal'] = analyze_tags_and_temporal(csv_file, 'data/tags_temporal_analysis.json')
    print(f"Completed in {time.time() - tags_start:.1f} seconds")
    
    # Generate comprehensive summary
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE SUMMARY...")
    print("="*80)
    
    summary = generate_summary(results)
    
    # Save combined results
    output_file = 'data/ACLED_COMPLETE_ANALYSIS.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    summary_file = 'ACLED_ANALYSIS_SUMMARY.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Complete analysis saved to: {output_file}")
    print(f"✓ Summary report saved to: {summary_file}")
    print(f"\nTotal analysis time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("="*80)
    
    return results

def generate_summary(results):
    """Generate a comprehensive markdown summary"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    summary = f"""# ACLED Data - Complete Analysis Summary

**Generated:** {timestamp}

---

## Dataset Overview

- **Total Events:** {results['geography']['total_events']:,}
- **Date Range:** January 1, 2024 - May 19, 2025
- **Countries Covered:** {results['geography']['total_countries']}
- **Regions Covered:** {results['geography']['total_regions']}
- **Unique Actor Types:** {results['actors']['unique_actor1_count']:,}

---

## 1. Event Type Distribution

### Disorder Types
"""
    
    for dtype in results['enums']['disorder_type']:
        summary += f"- {dtype}\n"
    
    summary += "\n### Event Types (5 categories)\n"
    for etype in results['enums']['event_type']:
        summary += f"- {etype}\n"
    
    summary += f"\n### Sub-Event Types ({len(results['enums']['sub_event_type'])} types)\n"
    for i, stype in enumerate(results['enums']['sub_event_type'], 1):
        summary += f"{i}. {stype}\n"
    
    summary += "\n---\n\n## 2. Geographic Distribution\n\n### By Region\n\n"
    summary += "| Region | Events | % of Total |\n"
    summary += "|--------|--------|------------|\n"
    
    total_events = results['geography']['total_events']
    for region in results['geography']['regions']:
        pct = (region['events'] / total_events) * 100
        summary += f"| {region['name']} | {region['events']:,} | {pct:.1f}% |\n"
    
    summary += "\n### Top 20 Countries\n\n"
    summary += "| Rank | Country | Region | Events |\n"
    summary += "|------|---------|--------|--------|\n"
    
    for i, country in enumerate(results['geography']['countries'][:20], 1):
        summary += f"| {i} | {country['name']} | {country['region']} | {country['events']:,} |\n"
    
    summary += "\n---\n\n## 3. Actor Analysis\n\n"
    
    summary += f"### Interaction Types ({len(results['actors']['interaction_types'])} types)\n\n"
    summary += "**Top 10 Interaction Patterns:**\n\n"
    summary += "| Interaction Type | Events |\n"
    summary += "|-----------------|--------|\n"
    
    for item in results['actors']['interaction_types'][:10]:
        summary += f"| {item['type']} | {item['count']:,} |\n"
    
    summary += "\n### Top 20 Primary Actors (Actor1)\n\n"
    summary += "| Rank | Actor | Events |\n"
    summary += "|------|-------|--------|\n"
    
    for i, actor in enumerate(results['actors']['top_100_actor1'][:20], 1):
        summary += f"| {i} | {actor['actor']} | {actor['count']:,} |\n"
    
    summary += "\n### Top 15 Base Actor Types\n\n"
    summary += "| Actor Type | Total Appearances |\n"
    summary += "|------------|------------------|\n"
    
    for actor in results['actors']['top_50_base_actor_types'][:15]:
        summary += f"| {actor['type']} | {actor['count']:,} |\n"
    
    summary += "\n---\n\n## 4. Temporal Patterns\n\n"
    
    if 'years' in results['tags_temporal']:
        summary += "### Events by Year\n\n"
        summary += "| Year | Events |\n"
        summary += "|------|--------|\n"
        for year in results['tags_temporal']['years']:
            summary += f"| {year['year']} | {year['events']:,} |\n"
    
    summary += "\n---\n\n## 5. Fatalities Analysis\n\n"
    
    if 'events_with_fatalities' in results['tags_temporal']:
        summary += f"- **Total Events:** {results['tags_temporal']['total_events']:,}\n"
        summary += f"- **Events with Fatalities:** {results['tags_temporal']['events_with_fatalities']:,}\n"
        summary += f"- **Total Fatalities:** {results['tags_temporal']['total_fatalities']:,}\n"
        summary += f"- **Average per Deadly Event:** {results['tags_temporal'].get('average_fatalities_per_deadly_event', 0)}\n"
        
        summary += "\n### Fatalities Distribution\n\n"
        summary += "| Range | Events |\n"
        summary += "|-------|--------|\n"
        
        for item in results['tags_temporal']['fatalities_distribution']:
            summary += f"| {item['range']} fatalities | {item['count']:,} |\n"
    
    summary += "\n---\n\n## 6. Data Quality Indicators\n\n"
    
    summary += "### Time Precision Levels\n\n"
    if 'time_precision' in results['tags_temporal']:
        for item in results['tags_temporal']['time_precision']:
            summary += f"- **Level {item['precision']}:** {item['count']:,} events\n"
    
    summary += "\n### Geographic Precision Levels\n\n"
    if 'geo_precision' in results['tags_temporal']:
        for item in results['tags_temporal']['geo_precision']:
            summary += f"- **Level {item['precision']}:** {item['count']:,} events\n"
    
    summary += "\n### Source Scale Distribution\n\n"
    summary += f"**{len(results['enums']['source_scale'])} source scale types:**\n\n"
    for scale in results['enums']['source_scale'][:10]:
        summary += f"- {scale}\n"
    
    summary += "\n---\n\n## 7. Interaction Categories\n\n"
    
    summary += f"### Actor Category 1 ({len(results['enums']['inter1'])} types)\n"
    for cat in results['enums']['inter1']:
        summary += f"- {cat}\n"
    
    summary += f"\n### Actor Category 2 ({len(results['enums']['inter2'])} types)\n"
    for cat in results['enums']['inter2']:
        summary += f"- {cat}\n"
    
    summary += f"\n### Interaction Patterns ({len(results['enums']['interaction'])} combinations)\n\n"
    summary += "See actor analysis section above for top patterns.\n"
    
    summary += "\n---\n\n## 8. Top Event Tags\n\n"
    
    if 'top_50_tags' in results['tags_temporal']:
        summary += "**Most Common Tags:**\n\n"
        summary += "| Tag | Frequency |\n"
        summary += "|-----|----------|\n"
        for item in results['tags_temporal']['top_50_tags'][:15]:
            summary += f"| {item['tag']} | {item['count']:,} |\n"
    
    summary += "\n---\n\n## Key Insights\n\n"
    
    # Calculate some insights
    top_country = results['geography']['countries'][0]
    top_region = results['geography']['regions'][0]
    protest_only = next((x for x in results['actors']['interaction_types'] if x['type'] == 'Protesters only'), None)
    
    summary += f"1. **Most Affected Country:** {top_country['name']} with {top_country['events']:,} events ({(top_country['events']/total_events*100):.1f}% of all events)\n"
    summary += f"2. **Most Active Region:** {top_region['name']} with {top_region['events']:,} events ({(top_region['events']/total_events*100):.1f}% of all events)\n"
    
    if protest_only:
        summary += f"3. **Peaceful Protests:** {protest_only['count']:,} 'Protesters only' events ({(protest_only['count']/total_events*100):.1f}% of all events)\n"
    
    if 'events_with_fatalities' in results['tags_temporal']:
        fatal_pct = (results['tags_temporal']['events_with_fatalities'] / total_events) * 100
        summary += f"4. **Violence Level:** {fatal_pct:.1f}% of events resulted in fatalities\n"
    
    summary += f"5. **Actor Diversity:** {results['actors']['unique_actor1_count']:,} unique primary actors tracked\n"
    summary += f"6. **Geographic Granularity:** {len(results['enums']['region'])} regions, {results['geography']['total_countries']} countries\n"
    
    summary += "\n---\n\n## Data Files Generated\n\n"
    summary += "- `data/enum_analysis.json` - Categorical field enumerations\n"
    summary += "- `data/country_analysis.json` - Geographic distributions\n"
    summary += "- `data/actor_analysis.json` - Actor patterns and interactions\n"
    summary += "- `data/tags_temporal_analysis.json` - Tags and temporal patterns\n"
    summary += "- `data/ACLED_COMPLETE_ANALYSIS.json` - Combined comprehensive analysis\n"
    summary += "\n---\n\n*Analysis completed using ACLED data from January 1, 2024 to May 19, 2025*\n"
    
    return summary

if __name__ == '__main__':
    run_all_analyses()

