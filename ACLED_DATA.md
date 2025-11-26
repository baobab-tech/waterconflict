# ACLED Data Documentation

## Overview
- **File**: `ACLED RAW_2024-01-01-2025-05-19.csv`
- **Size**: 235 MB
- **Total Records**: 483,634 events (excluding header)
- **Date Range**: January 1, 2024 - May 19, 2025

## Data Source
ACLED (Armed Conflict Location & Event Data Project) - a comprehensive real-time data and analysis source on political violence and protest around the world.

## Schema

The dataset contains 31 columns:

| Column Name | Description | Example Values |
|------------|-------------|----------------|
| `event_id_cnty` | Unique event identifier by country | PSE73898, YEM99796 |
| `event_date` | Date of the event | "16 May 2025" |
| `year` | Year of the event | 2025 |
| `time_precision` | Precision of the time data | 1 |
| `disorder_type` | Type of disorder/conflict | "Political violence", "Demonstrations" |
| `event_type` | Specific event type | "Protests", "Riots", "Battles", "Violence against civilians", "Explosions/Remote violence" |
| `sub_event_type` | Detailed event sub-category | "Peaceful protest", "Mob violence", "Armed clash", "Air/drone strike", "Shelling/artillery/missile attack" |
| `actor1` | Primary actor involved | "Protesters (Yemen)", "Military Forces of Russia (2000-)", "Rioters (Israel)" |
| `assoc_actor_1` | Associated actors with actor1 | "Government of Yemen (2017-) Houthi", "Settlers (Israel)" |
| `inter1` | Actor1 interaction category | "Protesters", "Rioters", "State forces", "External/Other forces" |
| `actor2` | Secondary actor involved | "Civilians (Palestine)", "Military Forces of Ukraine (2019-)" |
| `assoc_actor_2` | Associated actors with actor2 | "Farmers (Palestine)", "Civilians (Bolivia)" |
| `inter2` | Actor2 interaction category | "Civilians", "State forces" |
| `interaction` | Type of interaction between actors | "Rioters-Civilians", "State forces-External/Other forces", "Protesters only" |
| `civilian_targeting` | Whether civilians were targeted | "Civilian targeting", empty |
| `iso` | ISO country code | 275, 887, 68, 760 |
| `region` | Geographic region | "Middle East", "South America", "Europe", "South Asia" |
| `country` | Country where event occurred | "Palestine", "Yemen", "Bolivia", "Syria", "Ukraine", "Russia" |
| `admin1` | First administrative division | "West Bank", "La Paz", "Deir ez Zor", "Sumy" |
| `admin2` | Second administrative division | "Tubas", "Murillo", "Al Mayadin", "Sumskyi" |
| `admin3` | Third administrative division | Various sub-districts |
| `location` | Specific location name | "Al Farsiyah", "La Paz", "Al-Hawayij" |
| `latitude` | Geographic latitude | 32.3425, 13.9145, -16.4957 |
| `longitude` | Geographic longitude | 35.5114, 43.7869, -68.1336 |
| `geo_precision` | Geographic precision level | 1, 2 |
| `source` | Data source | "Palestine News and Information Agency", "Yemen News Agency (SABA) - Houthi" |
| `source_scale` | Scale of the source | "National", "Subnational-National", "Other", "Regional" |
| `notes` | Detailed description of the event | Full text description of what occurred |
| `fatalities` | Number of reported fatalities | 0, 1, 2, 10, etc. |
| `tags` | Event tags | "crowd size=no report", "crowd size=large" |
| `timestamp` | Unix timestamp | 1747690208 |

## Event Type Distribution

| Event Type | Count | Percentage |
|-----------|-------|------------|
| Protests | 205,033 | 42.4% |
| Explosions/Remote violence | 132,370 | 27.4% |
| Battles | 74,329 | 15.4% |
| Violence against civilians | 47,484 | 9.8% |
| Riots | 24,418 | 5.0% |

## Top 20 Countries by Event Count

| Country | Events |
|---------|--------|
| Ukraine | 77,978 |
| India | 34,049 |
| Palestine | 24,338 |
| Russia | 20,341 |
| Mexico | 19,354 |
| Myanmar | 18,491 |
| United States | 17,575 |
| Syria | 17,570 |
| Yemen | 14,383 |
| Lebanon | 14,324 |
| Brazil | 14,302 |
| Pakistan | 13,139 |
| Iraq | 10,073 |
| France | 9,296 |
| Turkey | 8,179 |
| South Korea | 8,146 |
| Sudan | 7,259 |
| Nigeria | 6,961 |
| Germany | 6,154 |
| Colombia | 6,008 |

## Sample Records

### Example 1: Political Violence in Palestine
```
Event ID: PSE73898
Date: 16 May 2025
Event Type: Riots - Mob violence
Location: Al Farsiyah, West Bank, Palestine
Actors: Israeli settler rioters vs Palestinian civilians
Description: Israeli settler rioters attacked a Palestinian shepherd with pepper spray as he was herding sheep.
Fatalities: 0
```

### Example 2: Demonstration in Yemen
```
Event ID: YEM99796
Date: 16 May 2025
Event Type: Protests - Peaceful protest
Location: Al Jillah, Ibb, Yemen
Actors: Protesters (Houthi-sponsored)
Description: Large Houthi-sponsored protest in solidarity with Palestinian people and support of Houthi actions.
Fatalities: 0
Tags: crowd size=large
```

### Example 3: Armed Clash in Bolivia
```
Event ID: BOL7391
Date: 16 May 2025
Event Type: Riots - Violent demonstration
Location: La Paz, Bolivia
Actors: Farmers/protesters vs Police Forces
Description: Farmers and Evo Morales supporters marched to register candidacy, police intervened with tear gas, resulting in clashes.
Fatalities: 0
Injuries: 12 (including 3 police, 1 elderly woman, 1 minor, 6 journalists)
Arrests: 5
Tags: crowd size=large
```

### Example 4: Military Action in Ukraine
```
Event ID: UKR227042
Date: 16 May 2025
Event Type: Explosions/Remote violence - Shelling/artillery/missile attack
Location: Hlushchenkove, Kharkiv, Ukraine
Actors: Russian military forces vs Ukrainian military forces
Description: Russian forces hit multiple Ukrainian units across several locations with artillery.
Fatalities: 2 (up to 225 reported by Russian sources)
```

### Example 5: Tribal Conflict in Pakistan
```
Event ID: PAK152305
Date: 16 May 2025
Event Type: Battles - Armed clash
Location: Birmal, South Waziristan, Pakistan
Actors: Sarki Khel Tribal Militia vs Jikhel-Darikhel Tribal Militia
Description: Clash between rival tribes over longstanding land dispute.
Fatalities: 1
Injuries: 2
```

## Data Characteristics

### Geographic Coverage
- Global dataset covering 100+ countries
- Strong focus on conflict zones and politically active regions
- Includes coordinates (latitude/longitude) for geographic analysis

### Event Types
1. **Demonstrations**: Protests, peaceful demonstrations
2. **Political Violence**: Armed conflicts, battles, violence against civilians
3. **Remote Violence**: Explosions, shelling, air strikes, drone attacks
4. **Riots**: Mob violence, violent demonstrations

### Temporal Coverage
- Daily event tracking from January 2024 to May 2025
- Real-time or near real-time updates

### Actor Information
- Detailed actor classification (state forces, rebel groups, civilians, protesters, etc.)
- Actor associations and affiliations
- Interaction types between actors

### Fatalities
- Most events (vast majority) have 0 fatalities
- Some high-fatality events in conflict zones (Ukraine, Syria, etc.)

## Use Cases

This dataset can be used for:
- Conflict analysis and monitoring
- Protest tracking and social movement research
- Geographic conflict mapping
- Actor network analysis
- Fatality and casualty analysis
- Regional stability assessment
- Early warning systems for violence
- Political event prediction
- Water conflict research (filtering events related to water resources)

