import os
import sqlite3
import xlwt 
from xlwt import Workbook 


wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1') 
student = []
presence = []
print("Files Available for exporting to pdf")
for each in os.listdir("./Database"):
    print(each)
print("Select a file from the list")
db_file = input()
conn = sqlite3.connect('./Database/'+db_file)
c = conn.cursor()
for each in c.execute('''SELECT * FROM record'''):
    student.append(each[0])
    presence.append(each[1])

sheet1.write(1,0,'Student')
sheet1.write(1,1,'Presence')
row_count = 2
for i in range(0,len(student)):
    sheet1.write(row_count,0,student[i])
    sheet1.write(row_count,1,presence[i])
    row_count = row_count+1
print("Enter name for the spreadsheet")
name = input()
wb.save('./Spreadsheets/'+name+'.xls') 