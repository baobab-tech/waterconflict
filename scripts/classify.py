"""
Simple classification script using the water-conflict-classifier model from Hugging Face Hub.

This script demonstrates:
- Loading the trained model from HF Hub
- Classifying a diverse set of 20 headlines
- Measuring inference timing
- Pretty-printing results
"""

import time
from setfit import SetFitModel

# Label names for the multi-label classifier
LABEL_NAMES = ["Trigger", "Casualty", "Weapon"]

# ============================================================================
# SAMPLE HEADLINES (20 mixed examples)
# ============================================================================
SAMPLE_HEADLINES = [
    # Water conflict examples (positive cases)
    "Taliban militants attacked workers at the Kajaki Dam construction site in southern Afghanistan, killing three engineers",
    "Israeli forces bombed water infrastructure in Gaza, leaving thousands without access to clean drinking water",
    "Armed groups seized control of the Mosul Dam in Iraq during intense fighting between government and insurgent forces",
    "Violent protests erupted in Bolivia over plans to privatize water systems, resulting in multiple casualties",
    "Syrian government forces targeted water pumping stations in rebel-held areas, causing humanitarian crisis",
    "Gunmen attacked a water treatment facility in Yemen, killing security guards and damaging critical infrastructure",
    "Clashes between farmers and herders in Mali over water access to irrigation canals left fifteen people dead",
    "Kurdish forces bombed Turkish-controlled dam structures along the Euphrates River in military operation",
    "Armed militants poisoned village water supplies in Nigeria during ethnic conflict, affecting hundreds of residents",
    "Police fired on protesters demonstrating against new dam project in Brazil, injuring dozens of indigenous activists",
    
    # Non-water conflict examples (negative cases)
    "New desalination plant opens in California to address drought conditions with innovative technology",
    "Scientists discover breakthrough water filtration method using graphene-based materials for purification",
    "City council approves budget for upgrading municipal water treatment systems to meet new standards",
    "Researchers report climate change impacts on rainfall patterns across Sub-Saharan African regions",
    "International water conservation conference brings together experts from fifty countries in Geneva",
    "Tech startup develops smart irrigation system that reduces agricultural water consumption by forty percent",
    "Marine biologists study coral reef ecosystems and their relationship to ocean water temperature changes",
    "Government announces new regulations for industrial wastewater discharge to protect river ecosystems",
    "Community celebrates completion of new well providing clean water access to rural village in Kenya",
    "Weather forecasts predict heavy monsoon rains and potential flooding in South Asian coastal regions"
]

def format_prediction(headline: str, prediction: list, index: int) -> str:
    """Format a single prediction result with labels."""
    labels_detected = [LABEL_NAMES[i] for i, val in enumerate(prediction) if val == 1]
    label_str = ", ".join(labels_detected) if labels_detected else "❌ No conflict"
    
    # Add emoji indicator
    emoji = "🔴" if labels_detected else "🟢"
    
    return f"{index:2d}. {emoji} {headline}\n    → Labels: {label_str}\n"

# ============================================================================
# MAIN CLASSIFICATION ROUTINE
# ============================================================================
def main():
    print("=" * 80)
    print("Water Conflict Classifier - Sample Classification Demo")
    print("=" * 80)
    print(f"Model: baobabtech/water-conflict-classifier")
    print(f"Headlines to classify: {len(SAMPLE_HEADLINES)}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model from Hugging Face Hub...")
    load_start = time.time()
    model = SetFitModel.from_pretrained("baobabtech/water-conflict-classifier")
    load_time = time.time() - load_start
    print(f"  ✓ Model loaded in {load_time:.2f}s")
    
    # Run inference
    print("\n[2/3] Running inference...")
    inference_start = time.time()
    predictions = model.predict(SAMPLE_HEADLINES)
    inference_time = time.time() - inference_start
    
    avg_time_per_headline = (inference_time / len(SAMPLE_HEADLINES)) * 1000  # Convert to ms
    print(f"  ✓ Classified {len(SAMPLE_HEADLINES)} headlines in {inference_time:.3f}s")
    print(f"  ✓ Average time per headline: {avg_time_per_headline:.1f}ms")
    
    # Display results
    print("\n[3/3] Results:")
    print("=" * 80)
    
    conflict_count = 0
    for i, (headline, pred) in enumerate(zip(SAMPLE_HEADLINES, predictions), start=1):
        print(format_prediction(headline, pred, i))
        if any(pred):
            conflict_count += 1
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total headlines classified: {len(SAMPLE_HEADLINES)}")
    print(f"Water conflict detected: {conflict_count} ({conflict_count/len(SAMPLE_HEADLINES)*100:.1f}%)")
    print(f"No conflict detected: {len(SAMPLE_HEADLINES) - conflict_count} ({(len(SAMPLE_HEADLINES) - conflict_count)/len(SAMPLE_HEADLINES)*100:.1f}%)")
    print(f"\nPerformance:")
    print(f"  - Model load time: {load_time:.2f}s")
    print(f"  - Total inference time: {inference_time:.3f}s")
    print(f"  - Average per headline: {avg_time_per_headline:.1f}ms")
    print(f"  - Throughput: {len(SAMPLE_HEADLINES)/inference_time:.1f} headlines/second")
    print("=" * 80)
    
    # Label breakdown
    label_counts = {label: 0 for label in LABEL_NAMES}
    for pred in predictions:
        for i, val in enumerate(pred):
            if val == 1:
                label_counts[LABEL_NAMES[i]] += 1
    
    if any(label_counts.values()):
        print("\nLabel Distribution in Detected Conflicts:")
        for label, count in label_counts.items():
            print(f"  - {label}: {count} occurrences")
        print("=" * 80)

if __name__ == "__main__":
    main()

