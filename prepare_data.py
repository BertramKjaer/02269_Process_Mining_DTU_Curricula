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

PRODUCE_CANONICAL_PATH = True  # If True, produce canonical dataset that follows obligatory course requirement strictly


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
if PRODUCE_CANONICAL_PATH:
    # Use course codes (KURSKODE) for strict matching. Normalize by removing
    # non-digit characters and leading zeros so '01901' and '1901' match.
    required_codes_raw = [
        "1901",
        "1904",
        "2313",
        "2315",
        "2312",
        "2326",
        "2327",
        "62409",
        "62577",
        "02324",
        "2323",
        "2332",
        "2369",
        "62588",
        "62550",
        "1920",
        "2346",
        "62410",
        "62999",
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


# %% Convert grading dates to sortable ISO 8601
df[GRADING_DATE] = pd.to_datetime(df[GRADING_DATE], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")


# %% Add semester column
def get_semester(value: str) -> str:
    date = pd.to_datetime(value, format="%Y-%m-%d")
    semester = "Fall" if date.month <= 4 or date.month >= 10 else "Spring"
    return f"{semester} {date.year}"


# %% Add COURSE_START column (one day before grading date)
df["COURSE_START"] = (pd.to_datetime(df[GRADING_DATE], format="%Y-%m-%d") - pd.Timedelta(days=1)).dt.strftime(
    "%Y-%m-%d"
)


# %% Sort data
df.sort_values(by=[STUDY_NUMBER, GRADING_DATE], inplace=True)


# %% Add attempt counter column
# This needs to be done after sorting by grading date
df["ATTEMPT"] = df.groupby([STUDY_NUMBER, COURSE_NUMBER]).cumcount() + 1


# %% Keep only relevant columns
df = df[
    [
        STUDY_NUMBER,
        COURSE_NUMBER,
        COURSE_TEXT,
        GRADE,
        "ATTEMPT",
        "COURSE_START",
        GRADING_DATE,
    ]
].copy()


# %% Export data
df.to_csv(OUTPUT_PATH, index=False)
