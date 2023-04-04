# sql1.py
"""Volume 1: SQL 1 (Introduction).
<Name> Trevor Wai
<Class> Section 1
<Date> 4/3/23
"""

import sqlite3 as sql
import csv
import matplotlib.pyplot as plt
import numpy as np

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn't exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of -1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    #Table Info
    with open(student_info, 'r') as infile:
        info_rows = list(csv.reader(infile))
    with open(student_grades, 'r') as infile:
        grades_rows = list(csv.reader(infile))
    major_rows = [(1, 'Math'), (2, 'Science'), (3, 'Writing'), (4, 'Art')]
    course_rows = [(1, 'Calculus'), (2, 'English'), (3, 'Pottery'), (4, 'History')]

    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            #Drops Tables if they exist
            cur.execute("DROP TABLE IF EXISTS MajorInfo;")
            cur.execute("DROP TABLE IF EXISTS CourseInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentGrades;")

            #Creates Tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT);")

            #Insert info into Tables
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", info_rows)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", grades_rows)
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", major_rows)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", course_rows)

            #Replace Values with NULL
            cur.execute("UPDATE StudentInfo SET MajorID=NUll WHERE MajorID==-1;")

    finally:
        conn.close()


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn't exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    #Earthquake Info
    with open(data_file, 'r') as infile:
        rows = list(csv.reader(infile))
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            #Drop Table if it Exists
            cur.execute("DROP TABLE IF EXISTS USEarthquakes;")

            #Creates Table
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL);")

            #Populate Table with Info
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)

            #Removes Earthquakes with Magnitude 0
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0")

            #Replaces 0 values with NULL values
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0")

    finally:
        conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Gets the list of students and courses that have an A or A+
            L = cur.execute("SELECT SI.StudentName, CI.CourseName "
                            "From StudentInfo AS SI, StudentGrades as SG, CourseInfo AS CI "
                            "WHERE (SG.Grade=='A+' or SG.Grade=='A') and SI.StudentID==SG.StudentID and CI.CourseID==SG.CourseID;").fetchall()
    finally:
        conn.close()

    return L

# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            #Obtain Information
            cent_19 = cur.execute("SELECT Magnitude From USEarthquakes WHERE Year>=1800 AND Year<1900;").fetchall()
            cent_20 = cur.execute("SELECT Magnitude From USEarthquakes WHERE Year>=1900 AND Year<2000;").fetchall()
            avg = cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;").fetchall()
    finally:
        conn.close()
    
    #19th Century plot
    plt.subplot(121)
    plt.hist(np.ravel(cent_19))
    plt.title("Magnitude of Earthequakes in the 19th Century")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    #20th Century plot
    plt.subplot(122)
    plt.hist(np.ravel(cent_20))
    plt.title("Magnitude of Earthquakes in the 20th Century")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return avg[0][0]
