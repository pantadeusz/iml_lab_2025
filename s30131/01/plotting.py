import matplotlib.pyplot as plt
import pandas as pd
file_path = "insurance.csv"
df = pd.read_csv(file_path)

srednie_bmi = df.groupby('age')['bmi'].mean().reset_index()

plt.plot(srednie_bmi['age'], srednie_bmi['bmi'])
plt.title('Średnia bmi do wieku')
plt.xlabel('Wiek')
plt.ylabel('Średnie Bmi')
plt.savefig('insurance.png')

