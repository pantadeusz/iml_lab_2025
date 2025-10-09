import matplotlib.pyplot as plt
import data_loading

plt.bar(data_loading.df1["age"][:10], data_loading.df1["pack_years"][:10])
plt.title("Plot")
plt.xlabel("age")
plt.ylabel("pack years")
plt.show()
plt.savefig("plot.png")
