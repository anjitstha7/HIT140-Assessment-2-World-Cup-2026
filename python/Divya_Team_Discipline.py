import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("datasets/Divya_Team_Discipline_FBref_Final_Population.csv")

# Clean the data
data = data.drop_duplicates("Team")
data = data[data["90s"] > 0]

# Yellow cards per 90 team-minutes
data["yellow_cards_90"] = data["Yellow_Cards"] / data["90s"]

# Separate the teams
knockout = data[data["Progression"] == "Knockout"]
eliminated = data[data["Progression"] == "Group-stage eliminated"]

# Select the sample
knockout = knockout.sample(16, random_state=42)
eliminated = eliminated.sample(8, random_state=42)

# Values used for the analysis
knockout_cards = knockout["yellow_cards_90"]
eliminated_cards = eliminated["yellow_cards_90"]

# Descriptive statistics
print("Descriptive Statistics")

print("\nKnockout Stage")
print(knockout_cards.describe().round(3))

print("\nGroup-stage Eliminated")
print(eliminated_cards.describe().round(3))


# 95% confidence interval
def get_confidence_interval(values):
    average = values.mean()
    margin = (
        stats.t.ppf(0.975, len(values) - 1)
        * stats.sem(values)
    )
    return average - margin, average + margin


knockout_ci = get_confidence_interval(knockout_cards)
eliminated_ci = get_confidence_interval(eliminated_cards)

print("\n95% Confidence Intervals")

print(
    "Knockout Stage:",
    round(knockout_ci[0], 3),
    "to",
    round(knockout_ci[1], 3)
)

print(
    "Group-stage Eliminated:",
    round(eliminated_ci[0], 3),
    "to",
    round(eliminated_ci[1], 3)
)


# Assumption checks
print("\nAssumption Checks")

print(
    "Knockout normality p:",
    round(
        stats.shapiro(knockout_cards).pvalue,
        4
    )
)

print(
    "Group-stage normality p:",
    round(
        stats.shapiro(eliminated_cards).pvalue,
        4
    )
)

print(
    "Levene p:",
    round(
        stats.levene(
            knockout_cards,
            eliminated_cards
        ).pvalue,
        4
    )
)


# Two-sample t-test
test = stats.ttest_ind(
    knockout_cards,
    eliminated_cards,
    equal_var=False
)

print("\nWelch Two-Sample T-Test")

print(
    "t-statistic:",
    round(test.statistic, 3)
)

print(
    "p-value:",
    round(test.pvalue, 4)
)


# Cohen's d
sd_knockout = knockout_cards.std()
sd_eliminated = eliminated_cards.std()

pooled_sd = np.sqrt(
    (
        (len(knockout_cards) - 1) * sd_knockout ** 2
        + (len(eliminated_cards) - 1) * sd_eliminated ** 2
    )
    /
    (
        len(knockout_cards)
        + len(eliminated_cards)
        - 2
    )
)

cohens_d = (
    knockout_cards.mean()
    - eliminated_cards.mean()
) / pooled_sd

print(
    "Cohen's d:",
    round(cohens_d, 3)
)


# Final decision
if test.pvalue < 0.05:
    print("Decision: Reject H0")
else:
    print("Decision: Fail to reject H0")


# Boxplot
plt.figure(figsize=(8, 5))

plt.boxplot(
    [knockout_cards, eliminated_cards],
    tick_labels=[
        "Knockout Stage",
        "Group-stage Eliminated"
    ]
)

plt.title("Yellow Cards per 90 Team-Minutes")
plt.xlabel("Tournament Progression")
plt.ylabel("Yellow Cards per 90 Team-Minutes")

plt.tight_layout()
plt.savefig(
    "figures/Divya_Team_Discipline_Boxplot.png",
    dpi=300
)

plt.show()


# Save the sample
sample = pd.concat(
    [knockout, eliminated],
    ignore_index=True
)

sample.to_csv(
    "datasets/Divya_Team_Discipline_Final_Sample.csv",
    index=False
)
