"""
Generate hard negative examples for water conflict classifier.

Hard negatives are water-related headlines that are NOT conflicts:
- Water infrastructure projects (peaceful)
- Water research and technology
- Water conservation initiatives
- Environmental water topics
- Water policy/regulations (non-violent)

These are critical for preventing false positives where the model
classifies any water-related news as a conflict.
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# HARD NEGATIVE TEMPLATES
# ============================================================================
# These are water-related but NON-conflict headlines

HARD_NEGATIVES = [
    # Water Infrastructure (peaceful development)
    "New desalination plant opens in California to address drought conditions with innovative technology",
    "City council approves $50 million budget for upgrading municipal water treatment systems",
    "Community celebrates completion of new well providing clean water access to rural village in Kenya",
    "State announces funding for water pipeline project to connect underserved communities",
    "Engineers complete construction of modern water purification facility in rural district",
    "Municipality unveils plans for new water storage reservoir to improve supply",
    "Regional water authority expands infrastructure to serve growing population needs",
    "Government inaugurates water treatment plant serving 100,000 residents in coastal city",
    "New pumping station improves water pressure for thousands of households in suburban area",
    "Water utility company completes upgrade of aging distribution network in downtown district",
    
    # Scientific Research & Technology
    "Scientists discover breakthrough water filtration method using graphene-based materials for purification",
    "Researchers develop new desalination technology that reduces energy costs by 40 percent",
    "Marine biologists study coral reef ecosystems and their relationship to ocean water temperature changes",
    "University team creates innovative sensor system for monitoring water quality in real-time",
    "Tech startup develops smart irrigation system that reduces agricultural water consumption significantly",
    "Study reveals microplastics in ocean water samples collected from remote Pacific locations",
    "Researchers identify bacteria species that could help purify contaminated water sources naturally",
    "Scientists use satellite data to track changes in groundwater levels across continent",
    "New water-saving technology promises to revolutionize industrial manufacturing processes",
    "Research team develops affordable filtration system for removing arsenic from drinking water",
    
    # Conservation & Environmental Management
    "International water conservation conference brings together experts from fifty countries in Geneva",
    "Environmental groups launch campaign to promote water-saving practices in drought-affected regions",
    "NGO distributes water-efficient irrigation systems to smallholder farmers in East Africa",
    "Wildlife conservation program focuses on protecting wetland habitats and water resources",
    "Community initiative promotes rainwater harvesting to reduce dependence on municipal supply",
    "Government announces water conservation measures during summer season to preserve resources",
    "Schools implement education program teaching students about water conservation importance",
    "Agricultural extension service trains farmers in drip irrigation and water management techniques",
    "City launches public awareness campaign about reducing household water consumption",
    "Environmental agency monitors river water quality to protect aquatic ecosystems",
    
    # Policy & Governance (non-violent)
    "Government announces new regulations for industrial wastewater discharge to protect river ecosystems",
    "Parliament passes legislation strengthening water quality standards for drinking supply",
    "Regional authorities establish water management committee to coordinate resource allocation",
    "Ministry releases national water policy framework for sustainable development goals",
    "Local council adopts water conservation bylaws for residential and commercial properties",
    "International treaty signed to cooperate on transboundary water resource management",
    "Environmental agency updates water pollution control regulations for manufacturing sector",
    "Government establishes water pricing reform to encourage efficient use and conservation",
    "Regulatory body issues guidelines for water utility companies to improve service delivery",
    "State legislature debates bill to fund rural water infrastructure improvements",
    
    # Weather & Climate
    "Weather forecasts predict heavy monsoon rains and potential flooding in South Asian coastal regions",
    "Meteorologists monitor tropical storm system bringing much-needed rainfall to drought areas",
    "Climate scientists project changes in precipitation patterns affecting water availability",
    "Seasonal forecasts indicate above-average rainfall expected during upcoming monsoon season",
    "Drought conditions ease following recent rainfall in water-stressed agricultural regions",
    "Meteorological agency issues flood watch for river basins due to sustained rainfall",
    "Climate report documents rising sea levels and impacts on coastal freshwater resources",
    "Weather service predicts dry season will extend longer than normal affecting water supplies",
    "Scientists track El Niño effects on rainfall distribution across Pacific region",
    "Hydrologists measure snowpack levels to forecast spring water runoff for reservoirs",
    
    # Agriculture & Irrigation (peaceful)
    "Farmers adopt modern irrigation techniques to maximize water efficiency in crop production",
    "Agricultural cooperative installs drip irrigation systems across member farmlands",
    "Extension workers demonstrate water-saving methods to improve farm productivity sustainably",
    "Crop scientists develop drought-resistant varieties requiring less water for cultivation",
    "Government subsidizes water-efficient irrigation equipment for small-scale farmers",
    "Farmers association shares best practices for managing water resources during dry spells",
    "Agricultural department promotes precision irrigation technology to reduce water waste",
    "Cooperative establishes water user group to manage shared irrigation canal system",
    "Research shows improved soil management helps retain water for longer growing seasons",
    "Training program teaches farmers to schedule irrigation based on crop water requirements",
    
    # Public Health (peaceful water initiatives)
    "Health ministry launches program to provide safe drinking water to remote communities",
    "Vaccination campaign includes water quality testing in areas affected by waterborne diseases",
    "NGO installs hand-washing stations with clean water at schools to improve hygiene",
    "Public health department monitors water sources for contamination to prevent disease outbreaks",
    "UNICEF partners with government to improve water and sanitation facilities in rural areas",
    "Health workers distribute water purification tablets to households in flood-affected regions",
    "Community health program trains volunteers in water treatment and hygiene education",
    "Medical research links improved water access to reduced incidence of diarrheal diseases",
    "Government initiative provides water filters to low-income families for safe drinking water",
    "Health officials conduct water quality assessments at public facilities including schools",
    
    # Business & Economics
    "Water utility company reports increased investment in infrastructure modernization",
    "Stock market: Water technology companies show strong growth in sustainable solutions sector",
    "Private sector invests in desalination projects to meet industrial water demands",
    "Water bottling company announces expansion of production facilities in three states",
    "Economic analysis shows water scarcity impacts agricultural productivity and GDP growth",
    "Venture capital firms increase funding for water innovation startups and technologies",
    "Beverage industry implements water stewardship programs to reduce consumption in manufacturing",
    "Water utility bonds attract investors seeking stable returns from essential services",
    "Market report: Global water treatment equipment industry projected to grow significantly",
    "Companies adopt water efficiency measures to reduce operational costs and environmental impact",
    
    # International Development & Aid
    "World Bank approves loan for water supply project in developing country",
    "International donors fund construction of water systems in refugee camps",
    "UN agency launches initiative to achieve universal access to safe drinking water by 2030",
    "Development bank finances rural water infrastructure projects across Africa",
    "Foreign aid program delivers water pumps and storage tanks to drought-affected communities",
    "Humanitarian organizations provide emergency water supplies following natural disaster",
    "International cooperation agreement supports transboundary water management projects",
    "Aid agencies coordinate water and sanitation interventions in conflict-affected regions",
    "Development partners invest in capacity building for water utility management",
    "International NGO completes water system rehabilitation in post-conflict area",
    
    # Education & Awareness
    "University establishes research center focused on water resources management",
    "Educational documentary explores global water challenges and innovative solutions",
    "Schools participate in water conservation competition promoting sustainable practices",
    "Online course teaches professionals about integrated water resources management",
    "Museum exhibition highlights importance of water in human civilization and development",
    "Youth groups organize clean-up campaign for local rivers and water bodies",
    "Workshop trains community leaders in water governance and stakeholder engagement",
    "Public library hosts lecture series on water history and future challenges",
    "Student science fair showcases projects related to water quality and conservation",
    "Media campaign raises awareness about water footprint of everyday products",
]

def generate_hard_negatives_csv(output_path: str = "../data/hard_negatives.csv"):
    """
    Generate CSV file with hard negative examples.
    
    Args:
        output_path: Path to save the hard negatives CSV
    """
    print("=" * 80)
    print("Hard Negatives Generator for Water Conflict Classifier")
    print("=" * 80)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Headline': HARD_NEGATIVES,
        'Basis': [''] * len(HARD_NEGATIVES)  # Empty basis for negatives
    })
    
    # Save to CSV
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated {len(HARD_NEGATIVES)} hard negative examples")
    print(f"✓ Saved to: {output_file.absolute()}")
    print(f"\nThese examples are water-related but NON-conflict headlines.")
    print(f"They should be merged with your existing negatives.csv for retraining.")
    print("\nNext steps:")
    print("  1. Review the generated hard negatives")
    print("  2. Merge with existing negatives.csv:")
    print("     - Read both CSVs")
    print("     - Concatenate dataframes")
    print("     - Save combined negatives.csv")
    print("  3. Retrain the model with updated dataset")
    print("  4. Re-evaluate with your classify.py test cases")
    print("=" * 80)
    
    return df


def merge_with_existing_negatives(hard_negatives_path: str = "../data/hard_negatives.csv",
                                  existing_negatives_path: str = "../data/negatives.csv",
                                  output_path: str = "../data/negatives_updated.csv"):
    """
    Merge hard negatives with existing negatives dataset.
    
    Args:
        hard_negatives_path: Path to hard negatives CSV
        existing_negatives_path: Path to existing negatives CSV
        output_path: Path to save merged negatives CSV
    """
    print("\n" + "=" * 80)
    print("Merging Hard Negatives with Existing Negatives")
    print("=" * 80)
    
    # Load both datasets
    hard_neg = pd.read_csv(hard_negatives_path)
    existing_neg = pd.read_csv(existing_negatives_path)
    
    print(f"\n✓ Loaded {len(hard_neg)} hard negatives (water-related, peaceful)")
    print(f"✓ Loaded {len(existing_neg)} existing negatives (general conflicts)")
    
    # Merge
    merged = pd.concat([existing_neg, hard_neg], ignore_index=True)
    
    # Remove duplicates if any
    original_count = len(merged)
    merged = merged.drop_duplicates(subset=['Headline'], keep='first')
    duplicates_removed = original_count - len(merged)
    
    if duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed} duplicate headlines")
    
    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_file = Path(output_path)
    merged.to_csv(output_file, index=False)
    
    print(f"\n✓ Merged dataset: {len(merged)} total negatives")
    print(f"  - Existing (general conflicts): {len(existing_neg)}")
    print(f"  - Hard negatives (peaceful water): {len(hard_neg)}")
    print(f"✓ Saved to: {output_file.absolute()}")
    print("\n⚠️  Review the merged file, then replace your negatives.csv with it")
    print("=" * 80)
    
    return merged


if __name__ == "__main__":
    # Generate hard negatives
    hard_neg_df = generate_hard_negatives_csv()
    
    # Ask user if they want to merge
    print("\n" + "=" * 80)
    response = input("Merge with existing negatives.csv now? (y/n): ").strip().lower()
    
    if response == 'y':
        merged_df = merge_with_existing_negatives()
        print("\n✅ Done! Now retrain your model with the updated negatives.")
    else:
        print("\n✓ Hard negatives saved. Merge manually when ready.")
        print("=" * 80)

