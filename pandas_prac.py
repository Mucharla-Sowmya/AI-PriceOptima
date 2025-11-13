#install pandas usnig pip install pandas
#import pandas
import pandas as pd 



#series
s=pd.Series([10,20,30,40])
print("series Data:\n")
print(s)
print("\n")



#dataframe
data = {
    'Name': ['Asha', 'Ravi', 'Kiran', 'Meena'],
    'Marks': [85, 90, 78, 88],
    'Subject': ['Math', 'Science', 'English', 'Physics']
}
df = pd.DataFrame(data)
print("Data in DataFrame:\n")
print(df)
print("\n")



#operations on df
print("First 5 rows of DataFrame:\n")
print(df.head())
print("\n")
print("Last 5 rows of DataFrame:\n")
print(df.tail())
print("\n")
print("Statistical Summary of DataFrame:\n")
print(df.describe())
print("\n")
print("DataFrame Info:\n")
print(df.info())
print("\n")
print("DataFrame Columns:\n")
print(df.columns)
print("\n")
print("DataFrame Shape:\n")
print(df.shape)
print("\n")



#-------Accessing data---------
print("Accessing 'Name' column:(single column)\n")
print(df['Name'])
print("\n")
print("Accessing 'Name' and 'Marks' columns:(multiple columns)\n")
print(df[['Name', 'Marks']])
print("\n")
print("Accessing row at index 1 using loc:\n")
print(df.loc[1])#loc is used to access a group of rows and columns (works on labels)-column name or row index
print("\n")
print("Accessing row at index 2 using iloc:\n")
print(df.iloc[2]) #iloc is used for integer-location based indexing for selection by position-row name and column index
print("\n")



#-------Filtering data---------
highscore = df[df['Marks'] > 80]
print("Students with Marks greater than 80:\n")
print(highscore)
print("\n")
print("Names and Marks of students with Marks greater than 80:\n")
print(df.loc[df['Marks'] > 80, ['Name', 'Marks']])
print("\n")

#-------Adding a new column---------
df['Grade'] = ['B', 'A', 'C', 'A']
print(df)
print("\n")
#-------Deleting a column---------
#df = df.drop('Subject', axis=1)
#print(df)
print("\n")
#-------Adding a new row---------
new_row = {'Name': 'Sita', 'Marks': 92, 'Subject': 'Chemistry'}
df.loc[len(df)] = new_row
print(df)
print("\n")
#-------Deleting a row---------
df = df.drop(2)  # Deleting row with index 2
print(df)
print("\n")


#-------Sorting data---------
sorted_df = df.sort_values(by='Marks', ascending=False)
print("DataFrame sorted by Marks in descending order:\n")
print(sorted_df)
print("\n")


#-------Grouping data---------
print("Average Marks by Subject:\n")
print(df.groupby('Subject')['Marks'].mean())
print("\n")

#-------Saving DataFrame to CSV---------
df.to_csv('students.csv', index=False)
print("DataFrame saved to 'students.csv'\n")
#-------Loading DataFrame from CSV---------
loaded_df = pd.read_csv('students.csv')
print("DataFrame loaded from 'students.csv':\n")
print(loaded_df)
print("\n")


#-------Basic Statistics Operations---------
print("Average Marks:", df['Marks'].mean())
print("Highest Marks:", df['Marks'].max())
print("Lowest Marks:", df['Marks'].min())
print("Total Marks:", df['Marks'].sum())
print("Standard Deviation of Marks:", df['Marks'].std())
print("Variance of Marks:", df['Marks'].var())
print("\n")

#-------Handling Missing Data---------
print(df.isnull())         # Shows True for missing values
print("\n")
print(df.fillna('A'))        # Replace missing with 'A'
print("\n")
r1 = {'Name': 'varun', 'Marks': 89, 'Subject': 'Chemistry'}
df.loc[len(df)] = r1
print(df)
print("\n")
print(df.dropna())         # Remove rows with missing data
print("\n")

#-------Merging DataFrames---------
df1 = pd.DataFrame({'Name': ['Asha', 'Ravi'], 'Marks': [85, 90]})
df2 = pd.DataFrame({'Name': ['Kiran', 'Meena'], 'Marks': [78, 88]})
merged_df = pd.concat([df1, df2])
print("Merged DataFrame:\n")
print(merged_df)
print("\n")