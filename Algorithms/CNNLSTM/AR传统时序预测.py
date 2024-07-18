import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
				 index_col=0, parse_dates=True)
print(df.head())
df.plot()
plt.show()

