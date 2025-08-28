import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import pandas as pd

# Load Data
#gdf = gpd.read_file("/Users/brookeengland/Documents/Internship/Project/Post-Processing/Intersect Operations/Labeled/all_years_labeled.gpkg")

# Average NDVI change per year
#plt.figure(figsize=(8, 5))
#sns.barplot(data=gdf, x="Year", y="ChangeNDVI", errorbar=None)
#plt.title("Average NDVI Change per Year")
#plt.ylabel("Mean Change in NDVI")
#plt.xlabel("Year")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

# Average Area change per year
#plt.figure(figsize=(8, 5))
#sns.barplot(data=gdf, x="Year", y="ChangeArea", errorbar=None)
#plt.title("Average Area Change per Year")
#plt.ylabel("Mean Change in Area")
#plt.xlabel("Year")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()


# Change in Area vs. Change in NDVI
#plt.figure(figsize=(8, 5))
#sns.scatterplot(data=gdf, x="ChangeNDVI", y="ChangeArea", hue="Status")
#plt.title("NDVI Change vs Area Change")
#plt.ylabel("Change in Area")
#plt.xlabel("Change in NDVI")
#plt.tight_layout()
#plt.show()

# 1. Load full dataset
gdf = gpd.read_file("/Users/brookeengland/Documents/Internship/Project/Post-Processing/Intersect Operations/Labeled/all_years_labeled.gpkg")

# 2. Filter out placeholder rows (optional)
gdf = gdf[gdf["Status"] != "missing"]

# 3. Get a list of unique SlipIDs with at least 3 years of data
eligible_slips = gdf.groupby("SlipID").filter(lambda x: x["Year"].nunique() >= 3)["SlipID"].unique()

# 4. Randomly sample 5 landslides
sampled_ids = pd.Series(eligible_slips).sample(5, random_state=42)  # Use random_state for reproducibility

# 5. Subset data for selected SlipIDs
sampled_data = gdf[gdf["SlipID"].isin(sampled_ids)]

# 6. Plot changes for each sampled landslide
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# NDVI Change
sns.lineplot(data=sampled_data, x="Year", y="ChangeNDVI", hue="SlipID", ax=axes[0, 0])
axes[0, 0].set_title("NDVI Change Over Time")

# Area Change
sns.lineplot(data=sampled_data, x="Year", y="ChangeArea", hue="SlipID", ax=axes[0, 1])
axes[0, 1].set_title("Area Change Over Time")

# Slope
if "Slope" in sampled_data.columns:
    sns.lineplot(data=sampled_data, x="Year", y="Slope", hue="SlipID", ax=axes[1, 0])
    axes[1, 0].set_title("Average Slope Over Time")
else:
    axes[1, 0].text(0.5, 0.5, "Slope data not available", ha="center")

# Aspect
if "Aspect" in sampled_data.columns:
    sns.lineplot(data=sampled_data, x="Year", y="Aspect", hue="SlipID", ax=axes[1, 1])
    axes[1, 1].set_title("Average Aspect Over Time")
else:
    axes[1, 1].text(0.5, 0.5, "Aspect data not available", ha="center")

plt.tight_layout()
plt.show()