The code folder contains scripts including: 

Advocacy_Cleaning_main.ipynb
⁃ Data cleaning and preparation pipeline. The dataset uploaded to OSF is not the raw data. We uploaded an anonymized dataset to protect the anonymity of participants. 
⁃ Data import: Remove redundant/uninformative columns. Merge demographic datasets we received from the sample provider and resolve discrepancies.
⁃ Relabel and transform variables: rename interventions, recode variables for clarity, create normalized variables, create composite variables.
⁃ Filtering: remove test data, duplicate entries, and participants who failed attention check or lacked unique identifiers.
⁃ Dataset exports: creates and exports a full cleaned dataset, a version without timing variables, and a version without intervention-specific columns.

Advocacy_Cleaning_time2.ipynb
⁃ Data cleaning steps (deleting test cases, calculating the completion time)

Advocacy_Descriptives.ipynb
⁃ Produces descriptive statistics and results discussed in supplementary materials, Section 1

Advocacy_Validations.Rmd
⁃ Produces results and tables included in supplementary materials, Section 2. Includes Bartlett’s and Levene’s tests, ANOVA analyses, mixed-effects models controlling for intervention duration, and models assessing the effect of a writing task.

Advocacy_Order_AS.R
⁃ Produces results included in supplementary materials, Section 2. Includes analyses testing for dependent variable order effects. 

Advocacy_Main.Rmd
⁃ Produces preregistered models. Includes mixed-effects models to assess intervention effects on advocacy outcomes, and False Discovery Rate calculation. Resulting tables are included in supplementary materials, Section 3.

Advocacy_IndOutcomes.Rmd
⁃ Produces tables included in supplementary materials, Section 4

Advocacy_Mediation.Rmd
⁃ Produces results and tables included in supplementary materials, Section 5

Advocacy_PartyModeration.Rmd
⁃ Produces results and tables included in supplementary materials, Section 6

Advocacy_OtherModerators.Rmd
⁃ Produces tables included in supplementary materials, Section 7

Advocacy_Attrition.ipynb
⁃ Produces results and tables included in supplementary materials, Section 8

Advocacy_Figs.ipynb
⁃ Produces figures included in the main text

Advocacy_Supp_Figs.ipynb
⁃ Produces figures included in supplementary materials

The data folder contains: 

advocacy_data.csv
⁃ The primary cleaned dataset. Anonymized dataset (with exclusions due to test data, duplicate entries, attention check failure, or lack of unique identifiers), without timers, and without intervention-related variables. 

data_anonymized.csv
⁃ Anonymized dataset (without exclusions due to test data, duplicate entries, attention check failure, or lack of unique identifiers). 

data_cleaned.csv
⁃ Anonymized dataset (with exclusions due to test data, duplicate entries, attention check failure, or lack of unique identifiers). 

data_notimers.csv
⁃ Anonymized dataset (with exclusions due to test data, duplicate entries, attention check failure, or lack of unique identifiers), and without the timers.

merged_data.csv
⁃ Cleaned datasets of time 1 and time 2 merged together.

codebook_advocacy.pdf
⁃ codebook for navigating the dataset 