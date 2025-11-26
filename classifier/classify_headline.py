from setfit import SetFitModel

# Load model
model = SetFitModel.from_pretrained("./water-conflict-classifier")

# Classify new headlines
headlines = [
    "Dam workers killed in attack",
    "New irrigation system installed"
]

predictions = model.predict(headlines)
# predictions = [[1, 1, 0], [0, 0, 0]]  # [Trigger, Casualty, Weapon]