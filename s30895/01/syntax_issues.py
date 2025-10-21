import sys
import pandas as pd
import matplotlib.pyplot as plt

data =  pd.read_csv("youtube-top-100-songs-2025.csv")
df = pd.DataFrame(data)

filtered_df = df[df[sys.argv[1]] >= int(sys.argv[2])]
filtered_df = filtered_df[filtered_df[sys.argv[1]] <= int(sys.argv[3])]

plt.hist(filtered_df[sys.argv[1]], bins=30)
plt.title(f'Histogram of {sys.argv[1]} (range: {sys.argv[2]}–{sys.argv[3]})')
plt.xlabel(sys.argv[1])
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("histogram.png",
            dpi=200,
            bbox_inches="tight"
            )
plt.show()
