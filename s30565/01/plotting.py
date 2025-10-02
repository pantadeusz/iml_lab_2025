import pandas as pd
import matplotlib.pyplot as plt

file_path = "Salary_Data.csv"
df = pd.read_csv(file_path)

avg_salary = df.groupby("Years of Experience")["Salary"].mean()

plt.plot(avg_salary.index, avg_salary, marker="o")
plt.title("Avg salary vs Years of experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.savefig("siema.png")

#dataset from https://www.kaggle.com/datasets/ayeshasiddiqa123/salary-data
