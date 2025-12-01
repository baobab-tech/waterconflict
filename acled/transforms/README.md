# ACLED Data Transform Scripts

Python scripts to analyze and extract unique values (enums) from the ACLED dataset.

## Scripts

### 1. `analyze_enums.py`
Extracts unique categorical values for key enum-like columns.

**Analyzes:**
- disorder_type
- event_type
- sub_event_type
- inter1 (actor1 interaction category)
- inter2 (actor2 interaction category)
- interaction (interaction between actors)
- civilian_targeting
- region
- source_scale

**Output:** `data/enum_analysis.json`

### 2. `analyze_countries.py`
Analyzes geographic distribution of events.

**Analyzes:**
- Countries and event counts
- Regions and event counts
- Country-to-region mappings
- ISO codes

**Output:** `data/country_analysis.json`

### 3. `analyze_actors.py`
Analyzes actor patterns and types.

**Analyzes:**
- Top actors (actor1 and actor2)
- Base actor types (extracted from full actor strings)
- Interaction types between actors
- Actor frequency patterns

**Output:** `data/actor_analysis.json`

### 4. `analyze_tags.py`
Analyzes tags and temporal patterns.

**Analyzes:**
- Event tags and frequencies
- Temporal distribution (years, months)
- Time and geographic precision levels
- Fatalities distribution
- Crowd size tags

**Output:** `data/tags_temporal_analysis.json`

## Usage

Run individual scripts:

```shell
# Analyze enum values
python3 transforms/analyze_enums.py

# Analyze geographic distribution
python3 transforms/analyze_countries.py

# Analyze actors
python3 transforms/analyze_actors.py

# Analyze tags and temporal patterns
python3 transforms/analyze_tags.py
```

Or run all at once:

```shell
cd transforms
for script in analyze_*.py; do
    echo "Running $script..."
    python3 "$script"
    echo ""
done
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## Output

All scripts generate:
1. **JSON files** in the `data/` directory with complete results
2. **Console output** with formatted summaries

The JSON files can be used for:
- Data validation
- Building dropdown/filter interfaces
- Understanding data distribution
- Creating data dictionaries

