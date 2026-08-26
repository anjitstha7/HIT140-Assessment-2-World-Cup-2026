import pandas as pd
import scipy.stats as st
import math
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. READ THE DATASET
# ------------------------------------------------------------

# Read the FIFA World Cup 2026 goalkeeper dataset.
# The first five rows contain FBref metadata/grouped headings.
df = pd.read_csv(
    "worldcup2026_goalkeepers.csv",
    skiprows=5
)


# ------------------------------------------------------------
# 2. DATA WRANGLING AND CLEANING
# ------------------------------------------------------------

# Rename the second Save% column because it represents
# penalty-kick save percentage.
df = df.rename(
    columns={"Save%.1": "PKSave%"}
)

# Remove FBref export/navigation columns that are not
# required for this analysis.
df = df.drop(
    columns=["Matches", "-9999"]
)

print("\nCleaned dataset dimensions:")
print(df.shape)


# ------------------------------------------------------------
# 3. DEFINE THE ELIGIBLE POPULATION
# ------------------------------------------------------------

# Include goalkeepers who played at least 90 minutes.
# This reduces the influence of extremely limited playing time
# on goalkeeper performance measures.
eligible_df = df[df["Min"] >= 90].copy()

print("\nEligible goalkeeper population:")
print(len(eligible_df))

print("\nMissing Save% values in eligible population:")
print(eligible_df["Save%"].isnull().sum())


# ------------------------------------------------------------
# 4. CREATE SHOOTING-PRESSURE VARIABLE
# ------------------------------------------------------------

# Calculate shots on target faced per 90 minutes.
# This standardises shooting pressure for different playing times.
eligible_df["SoTA_per90"] = (
    eligible_df["SoTA"] / eligible_df["90s"]
)

pressure_median = eligible_df["SoTA_per90"].median()

print("\nMedian SoTA per 90:")
print(pressure_median)


# ------------------------------------------------------------
# 5. CREATE SHOOTING-PRESSURE GROUPS
# ------------------------------------------------------------

# Goalkeepers below the median are classified as lower pressure.
lower_pressure = eligible_df[
    eligible_df["SoTA_per90"] < pressure_median
].copy()

# Goalkeepers above the median are classified as higher pressure.
higher_pressure = eligible_df[
    eligible_df["SoTA_per90"] > pressure_median
].copy()

print("\nLower-pressure population size:")
print(len(lower_pressure))

print("\nHigher-pressure population size:")
print(len(higher_pressure))


# ------------------------------------------------------------
# 6. STRATIFIED RANDOM SAMPLING
# ------------------------------------------------------------

# Randomly select 20 goalkeepers from each pressure group.
# random_state=42 makes the sampling reproducible.
lower_sample = lower_pressure.sample(
    n=20,
    random_state=42
)

higher_sample = higher_pressure.sample(
    n=20,
    random_state=42
)

# Combine both samples.
sample_df = pd.concat(
    [lower_sample, higher_sample],
    ignore_index=True
)

sample_df["Pressure_Group"] = (
    ["Lower Pressure"] * len(lower_sample)
    + ["Higher Pressure"] * len(higher_sample)
)

print("\nFinal sample size:")
print(len(sample_df))

print("\nLower-pressure sample size:")
print(len(lower_sample))

print("\nHigher-pressure sample size:")
print(len(higher_sample))


# ------------------------------------------------------------
# 7. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nSave% descriptive statistics - Lower Pressure:")
print(lower_sample["Save%"].describe())

print("\nSave% descriptive statistics - Higher Pressure:")
print(higher_sample["Save%"].describe())

mean_lower = lower_sample["Save%"].mean()
mean_higher = higher_sample["Save%"].mean()

std_lower = lower_sample["Save%"].std(ddof=1)
std_higher = higher_sample["Save%"].std(ddof=1)

n_lower = len(lower_sample)
n_higher = len(higher_sample)

mean_difference = mean_higher - mean_lower

print("\nMean Save% - Lower Pressure:")
print(mean_lower)

print("\nMean Save% - Higher Pressure:")
print(mean_higher)

print("\nDifference in mean Save% (Higher - Lower):")
print(mean_difference)


# ------------------------------------------------------------
# 8. 95% CONFIDENCE INTERVAL FOR DIFFERENCE IN MEAN SAVE%
# ------------------------------------------------------------

# Calculate the standard error for the difference between
# two independent sample means.
standard_error_difference = math.sqrt(
    (std_higher ** 2 / n_higher)
    +
    (std_lower ** 2 / n_lower)
)

# Calculate Welch-Satterthwaite degrees of freedom.
numerator = (
    (std_higher ** 2 / n_higher)
    +
    (std_lower ** 2 / n_lower)
) ** 2

denominator = (
    ((std_higher ** 2 / n_higher) ** 2 / (n_higher - 1))
    +
    ((std_lower ** 2 / n_lower) ** 2 / (n_lower - 1))
)

degrees_freedom = numerator / denominator

# Critical t-value for a two-sided 95% confidence interval.
t_critical = st.t.ppf(
    0.975,
    degrees_freedom
)

# Calculate margin of error.
margin_error_difference = (
    t_critical * standard_error_difference
)

# Calculate confidence interval.
difference_ci_lower = (
    mean_difference - margin_error_difference
)

difference_ci_upper = (
    mean_difference + margin_error_difference
)

print("\n95% Confidence Interval for Difference in Mean Save%")
print("(Higher Pressure - Lower Pressure)")

print("\nStandard error of difference:")
print(standard_error_difference)

print("\nWelch degrees of freedom:")
print(degrees_freedom)

print("\nMargin of error:")
print(margin_error_difference)

print("\n95% Confidence Interval:")
print(
    difference_ci_lower,
    "to",
    difference_ci_upper
)


# ------------------------------------------------------------
# 9. WELCH TWO-SAMPLE T-TEST
# ------------------------------------------------------------

# Null hypothesis (H0):
# Mean Save% is equal between the pressure groups.
#
# Alternative hypothesis (H1):
# Mean Save% differs between the pressure groups.

alpha = 0.05

t_statistic, p_value = st.ttest_ind(
    higher_sample["Save%"],
    lower_sample["Save%"],
    equal_var=False
)

print("\nWelch Two-Sample t-Test:")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print(
        "There is sufficient evidence of a statistically "
        "significant difference in mean Save% between "
        "the two pressure groups."
    )
else:
    print("Result: Fail to reject the null hypothesis.")
    print(
        "There is insufficient evidence of a statistically "
        "significant difference in mean Save% between "
        "the two pressure groups."
    )


# ------------------------------------------------------------
# 10. VISUALISATION
# ------------------------------------------------------------

plt.boxplot(
    [
        lower_sample["Save%"],
        higher_sample["Save%"]
    ],
    tick_labels=[
        "Lower Pressure",
        "Higher Pressure"
    ]
)

plt.title(
    "Goalkeeper Save Percentage by Shooting Pressure Group"
)
plt.xlabel("Shooting Pressure Group")
plt.ylabel("Save Percentage (%)")

plt.savefig(
    "goalkeeper_save_percentage_boxplot.png",
    bbox_inches="tight"
)

plt.show()