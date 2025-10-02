import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('BMW.csv')

plt.plot(df['Year'], df['Sales_Volume'])
plt.title('Przykładowy wykres')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()