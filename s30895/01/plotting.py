import matplotlib.pyplot as plt

import data_loading

df_limited = data_loading.df.head(5)


plt.plot(df_limited['title'], df_limited['view_count'])
plt.title('Youtube trending')
plt.xlabel('video title',)
plt.ylabel('view count (bln)')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(left=0.2,bottom=0.45, top=0.95)

plt.savefig("youtube_trending.png",
            dpi=200,
            bbox_inches="tight"
            )
plt.show()
