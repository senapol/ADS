Problem definition:
Using OSINT data in order to analyse the effect of military aid on the frontline of the Ukraine war
This report cleans, integrates, and explored key OSINT datasets relating to war and economic markers (i.e. SIPRI Milex, Arms Transfers, AGSI, AMECO, Eyes on Russia, ACLED)


# Part 1: Integrated war dataset (DATASET A)
Datasets used: ACLED, Eyes on Russia

An integrated dataset was made with collated event data related to the Ukraine war, sourced from validated social media posts (Eyes on Russia) and official military reports and media (ACLED).

Eyes on Russia: Obtained via the API endpoint in geoJSON format

ACLED: Downloaded from [link]

Preparation: Standardised 'event_type' across datasets to agree with ACLED definitions, as well as the dates. A new column was created to describe the 'source_dataset' (i.e. ACLED or Eyes on Russia)

Integrated dataset (combined_events.csv) has columns: date,latitude,longitude,event_type,location,admin1,description,source_dataset,source_id


Exploration consisted of...


# Part 2: Extracting the frontline
Several methods were used in order to extract the frontline from the latitude and longitude points of the conflict events stored in the integrated ACLED/EOR dataset.

Neural networks: using NN to classify the 'actor' and map the frontline accordingly between the 2 countries from the conflict data

Clustering: Using DBSCAN clustering with variable parameters (epsilon, minimum number of samples) allows us to eliminate noise and focus on dense areas of conflict to extract the frontline.

Regression: Regression without clustering led to a line unrepresentative of the frontline due to the 'noise' and unsorted points

Clustering-then-regression: Sorting the points and removing noise before implementing a regression technique seems to capture the movement of the frontline to a better extent than just clustering or just regression
    
    > Random Forest classifier: using the RF classifier to regress the points was not very successful
    > Polynomial regression
    > Parametric regression
    > Splines
    > PCA and principal curve fitting
    > Travelling salesman algorithm

Currently: PCA to reduce dimensionality before fitting has performed the best but is still not entirely representative or accurate

Smoothing: Savgol filter or spline smoothing

### Feature engineering: calculating area between 'frontline' and pre-war border

Border coordinates: obtained from [...] in JSON format

Polygon created between frontline and border, applied projection to the polygon to work out the area in sqkm, i.e. 'area gained by Russia'.

# Part 3: Integrating commidity and economic marker data (DATASET B)
Datasets used: AGSI (Aggregated Gas Storage Inventory), EIA (oil), Europa Oil Prices, AMECO (economic markers)

These datasets were merged and transformed from wide to long data, meaning the key index was the time unit.

This data is used to build our understanding of the economic side of the Ukraine war - i.e. we will be able to do correlation and causation analysis between this data and the war markers described in Dataset A.

    > Correlation methods:
    Pearson's R Coefficient
    Cohen's d
    ANOVA
    T-test

Testing correlation between the economic markers and 1) the number of events per year 2) change in frontline
OR the war markers with the military expenditure/militarty aid

# Part 4: Time series analysis
Time series modelling allows us to decompose data into seasonal, trend, and noise data. This will allow us to calculate the correlation without being affected by other large, omittable factors (i.e. the summer/winter cycle of oil expenditure).

Method: seasonal_decomposition function from the statsmodel library