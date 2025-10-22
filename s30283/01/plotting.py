import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
plt.hist(df['RhythmScore'], bins=80, color='green')
plt.title('Rhythm Scores')
plt.xlabel('Rhythm Score')
plt.ylabel('Count')
plt.savefig('chart.png')
plt.show()