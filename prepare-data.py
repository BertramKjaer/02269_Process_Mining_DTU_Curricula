#!/usr/bin/env python3

# %% Declare constants
INPUT_PATH = "DTU_Curricula_Data.csv"
OUTPUT_PATH = "DTU_Curricula_Data_Filtered.csv"

STUDY_NUMBER = "STUDIENR"
EDUCATION = "UDDANNELSE"
COURSE_NUMBER = "KURSKODE"
COURSE_TEXT = "KURSTXT"
GRADE = "BEDOMMELSE"
GRADING_SCALE = "SKALA"
ECTS = "ECTS"
EXAM_FORM = "UDPROVNING"
CENSOR = "CENSUR"
GRADING_DATE = "BEDOMMELSESDATO"

# %% Load data
import pandas as pd

df = pd.read_csv(INPUT_PATH, sep=";")

# %% Keep only students who study "Softwareteknologi, ing.prof.bach."
df = df[df[EDUCATION] == "Softwareteknologi, ing.prof.bach."]
df.drop(columns=[EDUCATION], inplace=True)

# %% Remove all rows with study numbers that only has two or fewer courses
course_counts = df[STUDY_NUMBER].value_counts()
valid_study_numbers = course_counts[course_counts > 2].index
df = df[df[STUDY_NUMBER].isin(valid_study_numbers)]

# %% Remove all rows with study numbers that has courses with "institut" in the course text
institute_study_numbers = df.loc[
    df[COURSE_TEXT].str.contains("institut", regex=False, na=False, case=False),
    STUDY_NUMBER,
].unique()
df = df[~df[STUDY_NUMBER].isin(institute_study_numbers)]

# %% Standardize course text by converting to uppercase and removing whitespace, diacritics and leading course number
import re

from unidecode import unidecode


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"^\d+", "", text)
    text = unidecode(text)

    return text.strip()


df[COURSE_TEXT] = df[COURSE_TEXT].apply(normalize_text)

# %% Normalize duplicate courses that has the same course text and ECTS but different course numbers

# For groups with the same course text and ECTS but multiple course numbers,
# pick the most frequent course number and assign it to the whole group.
grouped = df.groupby([COURSE_TEXT, ECTS])[COURSE_NUMBER]
counts = grouped.nunique()
duplicate_keys = counts[counts > 1].index  # MultiIndex of (COURSE_TEXT, ECTS)

if len(duplicate_keys) > 0:
    canonical = grouped.apply(lambda s: s.value_counts().idxmax())

    mapping = {key: canonical.loc[key] for key in duplicate_keys}

    keys = list(zip(df[COURSE_TEXT], df[ECTS]))
    df[COURSE_NUMBER] = [mapping.get(k, num) for k, num in zip(keys, df[COURSE_NUMBER])]

    # Optional: drop exact duplicate rows that may appear after normalization
    df.drop_duplicates(inplace=True)

# %% Remove all rows with study numbers that has failed a course
# failed_ratings = {"-3", "0", "EM", "IG", "IB"}
# failed_study_numbers = df.loc[df[GRADE].isin(failed_ratings), STUDY_NUMBER].unique()
# df = df[~df[STUDY_NUMBER].isin(failed_study_numbers)]

# %% Convert grading dates to sortable ISO 8601
df[GRADING_DATE] = pd.to_datetime(df[GRADING_DATE], format="%d/%m/%Y")
df[GRADING_DATE] = df[GRADING_DATE].dt.strftime("%Y-%m-%d")


# %% Add semester column
def get_semester(value: str) -> str:
    date = pd.to_datetime(value, format="%Y-%m-%d")
    semester = "Fall" if date.month <= 4 or date.month >= 10 else "Spring"
    return f"{semester} {date.year}"


df["SEMESTER"] = df[GRADING_DATE].apply(get_semester)


# %% Set the grading date of courses in the same semester to the same date
def set_semester_grading_dates(group: pd.DataFrame) -> pd.DataFrame:
    semester = group["SEMESTER"].iloc[0]
    year = pd.to_datetime(group[GRADING_DATE]).dt.year
    if "Spring" in semester:
        new_date = pd.Timestamp(year=year.iloc[0], month=6, day=1)
    else:
        new_date = pd.Timestamp(year=year.iloc[0], month=12, day=1)

    group["SEMESTER_END"] = new_date.strftime("%Y-%m-%d")
    return group


df = df.groupby([STUDY_NUMBER, "SEMESTER"]).apply(set_semester_grading_dates)
df.reset_index(drop=True, inplace=True)


# %% Add attempt counter column

df["ATTEMPT"] = df.groupby([STUDY_NUMBER, COURSE_NUMBER]).cumcount() + 1

# %% Sort data
df.sort_values(by=[STUDY_NUMBER, COURSE_NUMBER, "SEMESTER_END"], inplace=True)

# %% Export data
df.to_csv(OUTPUT_PATH, index=False)
