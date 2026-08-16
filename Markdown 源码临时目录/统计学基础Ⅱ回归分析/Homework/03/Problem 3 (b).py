import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Data reorganization for two-way ANOVA
# Temperature: [10, 10, 55, 55] -> indexed as [1, 1, 2, 2]
# Concentration: [20, 40, 20, 40] -> indexed as [1, 2, 1, 2]
data = {
    "Absorption": [0.28, 0.278, 0.38, 0.392, 0.266, 0.258, 0.332, 0.334],
    "Temperature": [10, 10, 10, 10, 55, 55, 55, 55],
    "Concentration": [20, 20, 40, 40, 20, 20, 40, 40],
}

# Create dataframe
df = pd.DataFrame(data)

# Fit the two-way ANOVA model with interaction
model = ols("Absorption ~ C(Temperature) * C(Concentration)", data=df).fit()
anova_results = anova_lm(model)

# Extract parameter estimates
parameter_estimates = model.params

anova_results, parameter_estimates
