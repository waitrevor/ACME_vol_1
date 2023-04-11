# solutions.py
"""Volume 1: SQL 2.
<Name> Trevor Wai
<Class> Section 1
<Date> 4/5/23
"""

import sqlite3 as sql

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    try:
        with sql.connect(db_file) as conn:
            current = conn.cursor()

            #Finds the specified student names
            ans = current.execute("SELECT SI.StudentName "
                                  "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                                  "ON SI.StudentID == SG.StudentID "
                                  "WHERE SG.Grade == 'B'").fetchall()
            
    finally:
        conn.commit()
        conn.close()

    return [val[0] for val in ans]


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            current = conn.cursor()

            #Finds the name majorname and grade of the students
            ans = current.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                                  "FROM StudentInfo AS SI LEFT OUTER JOIN StudentGrades AS SG "
                                  "ON SI.StudentID == SG.StudentID "
                                  "LEFT OUTER JOIN MajorInfo as MI "
                                  "On SI.MajorID == MI.MajorID "
                                  "WHERE SG.CourseID == 1;").fetchall()
            
    finally:
        conn.commit()
        conn.close()

    return ans


# Problem 3
def prob3(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            current = conn.cursor()

            #Finds the number of students in a specified major name and sorts them
            ans = current.execute("SELECT MI.MajorName, COUNT(*) AS N "
                                  "FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo AS MI "
                                  "ON SI.MajorID == MI.MajorID "
                                  "GROUP BY SI.MajorID "
                                  "ORDER BY N DESC, SI.StudentName ASC;").fetchall()
            
    finally:
        conn.commit()
        conn.close()

    return ans


# Problem 4
def prob4(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            current = conn.cursor()

            #Find the Student and GPA
            ans = current.execute("SELECT SI.StudentName, COUNT(*), AVG(SG.gradepoint) as gpa "
                                  "FROM ("
                                  "SELECT StudentID, CASE Grade "
                                  "WHEN 'A+' THEN 4.0 "
                                  "WHEN 'A' THEN 4.0 "
                                  "WHEN 'A-' THEN 3.7 "
                                  "WHEN 'B+' THEN 3.4 "
                                  "WHEN 'B' THEN 3.0 "
                                  "WHEN 'B-' THEN 2.7 "
                                  "WHEN 'C+' THEN 2.4 "
                                  "WHEN 'C' THEN 2.0 "
                                  "WHEN 'C-' THEN 1.7 "
                                  "WHEN 'D+' THEN 1.4 "
                                  "WHEN 'D' THEN 1.0 "
                                  "WHEN 'D-' THEN 0.7 "
                                  "END AS gradepoint "
                                  "FROM StudentGrades) AS SG "
                                  "INNER JOIN StudentInfo AS SI "
                                  "ON SG.StudentID == SI.StudentID "
                                  "GROUP BY SG.StudentID "
                                  "ORDER BY gpa DESC;").fetchall()
    finally:
        conn.commit()
        conn.close()

    return ans


# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """
    try:
        with sql.connect(db_file) as conn:
            current = conn.cursor()

            #Find the outliers name, id number, eyecolor and height
            id = current.execute("SELECT ID_number FROM table_2 WHERE description LIKE '%Alaska%';").fetchall()[0][0]
            name = current.execute("SELECT name FROM table_1 WHERE name LIKE '%William T.%';").fetchall()[0][0]
            ans = current.execute("SELECT eye_color, height FROM table_3 WHERE eye_color == 'Hazel-blue';").fetchall()[0]

    finally:
        conn.commit()
        conn.close()

    return [name, id, *ans]
