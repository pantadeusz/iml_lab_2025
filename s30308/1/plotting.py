import matplotlib.pyplot as plt

from data_loading import df

# Przykład wykresu liniowego
plt.plot(df['hours_studied'], df['sleep_hours'])
plt.title('Przykładowy wykres')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.savefig('student_exam_scores.png')