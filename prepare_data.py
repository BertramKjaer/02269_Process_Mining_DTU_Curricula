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
print(f"Students after education filter: {df[STUDY_NUMBER].nunique()}")

# %% Standardize course text by converting to uppercase and removing whitespace, diacritics and leading course number
import re

from unidecode import unidecode


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"^\d+", "", text)
    text = unidecode(text)

    return text.strip()


df[COURSE_TEXT] = df[COURSE_TEXT].apply(normalize_text)

# %% Keep only students who have taken all obligatory courses (by course code)
# Use course codes (KURSKODE) for strict matching. Normalize by removing
# non-digit characters and leading zeros so '01901' and '1901' match.
required_codes_raw = [
    "1901","1904","2313","2315","2312","2326","2327","62409","62577","02324","2323","2332","2369","62588","62550","1920","2346","62410","62999"
]

def clean_code(code: str) -> str:
    if pd.isna(code):
        return ""
    s = re.sub(r"\D", "", str(code))
    return s.lstrip("0")

required_codes = {clean_code(c) for c in required_codes_raw}

# Add cleaned code column to dataframe
df["COURSE_CODE_CLEAN"] = df[COURSE_NUMBER].apply(clean_code)

# Group codes per student and keep only students who have all required codes
codes_by_student = df.groupby(STUDY_NUMBER)["COURSE_CODE_CLEAN"].apply(lambda s: set([x for x in s if x]))
before_students = df[STUDY_NUMBER].nunique()
valid_students_codes = codes_by_student[codes_by_student.apply(lambda s: required_codes.issubset(s))].index
after_students = len(valid_students_codes)
print(f"Students matching obligatory course codes: {after_students} (from {before_students})")
df = df[df[STUDY_NUMBER].isin(valid_students_codes)]

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


# Assign `SEMESTER_END` explicitly per (STUDIENR, SEMESTER) group to avoid
# pandas groupby.apply behavior that can place grouping keys into the index
# and make columns unavailable for later operations.
df["SEMESTER_END"] = pd.NA
for (studnr, sem), grp in df.groupby([STUDY_NUMBER, "SEMESTER"]):
    year = pd.to_datetime(grp[GRADING_DATE]).dt.year.iloc[0]
    if "Spring" in sem:
        new_date = pd.Timestamp(year=year, month=6, day=1)
    else:
        new_date = pd.Timestamp(year=year, month=12, day=1)

    df.loc[grp.index, "SEMESTER_END"] = new_date.strftime("%Y-%m-%d")

# %% Add semester start date column


def get_semester_start(value: str) -> str:
    date = pd.to_datetime(value, format="%Y-%m-%d")
    if date.month == 6:
        start_date = pd.Timestamp(year=date.year, month=2, day=1)
    else:
        start_date = pd.Timestamp(year=date.year, month=8, day=1)
    return start_date.strftime("%Y-%m-%d")


df["SEMESTER_START"] = df["SEMESTER_END"].apply(get_semester_start)


# %% Add attempt counter column

df["ATTEMPT"] = df.groupby([STUDY_NUMBER, COURSE_NUMBER]).cumcount() + 1


# %% Combine courses with attempts into single rows with adjusted end dates based on number of attempts
# Convert semester start/end to datetime so min/max aggregation is chronological
df["SEMESTER_START"] = pd.to_datetime(df["SEMESTER_START"], format="%Y-%m-%d")
df["SEMESTER_END"] = pd.to_datetime(df["SEMESTER_END"], format="%Y-%m-%d")

aggregation_functions = {
    COURSE_TEXT: "first",
    GRADE: "last",
    GRADING_SCALE: "first",
    ECTS: "first",
    EXAM_FORM: "first",
    CENSOR: "first",
    # Keep semester listing as joined string; ordering depends on original row order
    "SEMESTER": lambda x: ", ".join(x),
    # Use min/max on datetimes so start is earliest and end is latest
    "SEMESTER_START": "min",
    "SEMESTER_END": "max",
    # ATTEMPT should reflect the highest attempt number
    "ATTEMPT": "max",
}

df = df.groupby([STUDY_NUMBER, COURSE_NUMBER], as_index=False).agg(aggregation_functions)

# Calculate end date based on semester start and number of attempts (one semester is 13 weeks)
df["SEMESTER_END"] = df["SEMESTER_START"] + pd.to_timedelta(df["ATTEMPT"] * 13 * 7, unit="days")

# After aggregation and adjustment, convert semester start/end back to ISO date strings
df["SEMESTER_START"] = pd.to_datetime(df["SEMESTER_START"]).dt.strftime("%Y-%m-%d")
df["SEMESTER_END"] = df["SEMESTER_END"].dt.strftime("%Y-%m-%d")
# %% Sort data
df.sort_values(by=[STUDY_NUMBER, "SEMESTER_END", COURSE_NUMBER], inplace=True)

# %% Export data
df.to_csv(OUTPUT_PATH, index=False)
