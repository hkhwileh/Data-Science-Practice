import warnings

import pandas as pd

warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\Hassan\Desktop\ML I\dataset\Student Mental health.csv")
print(data.head())
# check last five rows of the dataset
print(data.tail())
# check shape of the dataset

print(data.shape)

# check mathamatic realstion ship of the dataset
print(data.describe())
# check out the correlation for te dataset

print(data.corr())
print(data['Choose your gender'].count())

# data cleaning
data = data[data["Age"].notna()]
print(data.shape)
data.columns = ['Timestamp', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 'Martial_Status', 'Depression', 'Anxiety',
                'Pain_Attach', 'Specialist_Treatment']
print(data['Year'].unique())


def Clean(Text):
    Text = Text[-1]
    Text = int(Text)
    return Text


data['Year'] = data['Year'].apply(Clean)
print("First three values of Year after cleaning text:")
print(data["Year"][:3], "\n")

# CGPA
print(data['CGPA'].unique())


def remove_space(string):
    string = string.strip()
    return string


data['CGPA'] = data['CGPA'].apply(remove_space)

print("First three values of CGPA after cleaning text:")
print(data["CGPA"][:3], "\n")
print(data["CGPA"].unique())

data2 = data.columns
print(data2.shape)


# We can observe that a lot of courses are interpreted differently though they mean the
# same, so we need to take care of that.
def replace(  # type: ignore[override]
        self,
        to_replace=None,
        value=lib.no_default,
        inplace: bool = False,
        limit: int | None = None,
        regex: bool = False,
        method: Literal["pad", "ffill", "bfill"] | lib.NoDefault = lib.no_default,
) -> Series | None:


    data['Course'].replace()
