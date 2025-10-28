import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flight_data_2024.csv")
loty_na_miesiac = df['month'].value_counts().sort_index()

plt.plot(loty_na_miesiac.index, loty_na_miesiac.values)
plt.title('Liczba lotów w poszczególnych miesiącach (2024)')
plt.xlabel('Miesiąc')
plt.ylabel('Liczba lotów')
plt.savefig("liczba_lotow.png", dpi=300)
plt.show()
