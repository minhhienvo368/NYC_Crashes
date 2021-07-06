# NYC_Crashes
This is 2nd assignment about Data cleaning
![](https://images.nycgo.com/image/fetch/q_70,w_900/https://www.nycgo.com/images/uploads/NY_in_3_days/TimeSquare-Manhattan-NYC-BrittanyPetronella_0069sat.jpg)

## The Mission

Bill de Blasio, mayor of New York City, is in a bit of a pickle. Indeed, his police department, the NYPD, collected information about all the traffic accidents that happened in New York City. However, they are too busy eating doughnuts to correctly encode each traffic incident, and so it happens that the dataset that we got here is quite dirty, has a lot of missing values and can't be used by a machine learning model as is.
Can you help Mr. de Blasio and shine a new light on his police department ?

What he wants exactly is to predict which streets are the most dangerous while visiting the [city that never sleeps](https://en.wikipedia.org/wiki/The_City_That_Never_Sleeps).

### Must-have features

- The dataset contains no missing values ("" or null)
- No duplicates.
- Values are consolidated
- Data format is correct
- No blank spaces (ex: `" I love python "` => `"I love python"`)

### Nice-to-have features

- The more rows of data you use, the better. However, pay attention that the more data you have, the longer each operation needs to execute.
- Add new features computed using the features present that you think are going to be useful.
- Apply the preprocessing steps needed so that a future machine learning model can make the best use out of it **(feature selection, feature engineering, feature normalization, and resampling)**

Pimp up in this README file:

# Description
1. Adding new parameters to keep #crash_date_time as its dtype, 
df_orig = pd.read_csv(filename, parse_dates=[['crash_date', 'crash_time']], skipinitialspace = True)

2. Learnt how to split the feature crash_date_time into day, month and year
df["day"] = df['crash_date_time'].map(lambda x: x.day)
df["month"] = df['crash_date_time'].map(lambda x: x.month)
df["year"] = df['crash_date_time'].map(lambda x: x.year)
df["hour"] = df['crash_date_time'].map(lambda x: x.hour)
df["minute"] = df['crash_date_time'].map(lambda x: x.minute)

df[['crash_date_time', 'hour', 'minute','day', 'month', 'year']].head()

3. Checked for unique of data (cols): There are no duplicates
df.duplicated().any()
4. Checked: how many rows of each attribute are NaN
print(f'Sum of null values by columns: {df.isnull().sum()} \n')

5. Drop the redundant columns that do not need for target and machine learning
6. Checked again the redundant spaces in each cell of dateframe
7. Rename unconventional column name
8. Consolidate the text values (converted to lowercase)
9. 



----------
Bill de Blasio wants to know and predict the dangerousness of the streets in New York.

- Ask yourself what your target variable will be.

  - Deaths/injuries/both ?
  - How about passengers/driver/pedestrians/cyclists ?

During the cleaning process, you might ask yourself some questions like:

- _When should I drop my column when I have missing data? 1. columns contain than 5%, 30%, >50% missing values? No...it depends
- _What do I do if I can't fill in every missing value ? 
- _How long should my cleaning process take to run ?_

And the response is: **it depends.**
Indeed, it depends on the precision that you need.

If Bill de Blasio just wants a quick and dirty version, and you only have 2 days, well, have a quick dirty version ready as quick as possible.
Bill de Blasio gives you 2 weeks to have a very precise dataset ? Well, you can invest more time into getting every last bit of data cleaned.

One advice: **Start small, then be more precise if you have more time.**
You can have multiple final cleaned datasets, like:

1. `data_clean_GOOD_ENOUGH.csv`
2. `data_clean_GOOD.csv`
3. `data_clean_PRECISE.csv`

## Evaluation criteria

| Criteria       | Indicator                                                                      | Yes/No |
| -------------- | ------------------------------------------------------------------------------ | ------ |
| 1. Is complete | The student has realized all must-have features.                               |        |
|                | There is a published GitHub page available.                                    |        |
| 2. Is Correct  | There are no warnings or errors when running the script.                       |        |
|                | The final `.csv` file can be opened and is in a clean state.                   |        |
|                | The learner has used functions and minimized code duplication.                 |        |
|                | The learner can explain why and how he replaced missing values.                |        |
| 3. Is great    | Significantly more than 100000 rows have been used.                            |        |
|                | Features were normalized and categorical variables have been encoded properly. |        |
