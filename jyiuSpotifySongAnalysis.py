# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# %%
df = pd.read_csv("audiodata.csv")
df = df.dropna(axis = 0, how = 'any')
df

# %%
# Convert data rows into floats
datacols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'time_signature', 'key']
for i in range(len(datacols)) :
    df[datacols[i]].astype(float)

# %%
df #.sort_values(['instrumentalness'], ascending=True)

# %%
# Correlation between danceability and mood
x = df["tempo"].values
y = df["valence"].values

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

regr = linear_model.LinearRegression()
regr.fit(x,y)

fig = plt.figure(figsize=(6, 6))
fig.suptitle("Tempo vs. Valence")

ax = plt.subplot(1, 1, 1)
ax.scatter(x, y, alpha=0.5)
ax.plot(x, regr.predict(x), color="red", linewidth=3)
plt.xticks(())
plt.yticks(())

ax.set_ylim(0,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(20.0))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(10.0))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

plt.xlabel("Tempo")
plt.ylabel("Valence")

plt.show()

# %%
# plt.ylim(0, 1)
# # sns.set(rc={'figure.figsize' : (15,15)})
# sns.regplot(x='loudness', y='energy', data = df, line_kws= {'color':'red'}).set(title="Loudness vs. Energy")

# %%
acousticness = 'acousticness'
danceability = 'danceability'
energy = 'energy'
instrumentalness = 'instrumentalness'
liveness = 'liveness'
loudness = 'loudness'
speechiness = 'speechiness'
tempo = 'tempo'
valence = 'valence'

fig = plt.hist(df[valence])

plt.xlabel(valence)
plt.ylabel('frequency')

plt.show()

# %%
x = "danceability"
y = "valence"

fig, (ax1, ax2), (ax3, ax4) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10, 5))
fig.suptitle("Histograms Showing Danceability and Valence Spread")
h = ax2.hist2d(df[x], df[y], bins=20)
ax1.hist(df["energy"])

ax2.set_xlabel(x)
ax2.set_ylabel(y)

ax1.set_xlabel("energy")

plt.colorbar(h[3], ax=ax2)

plt.show()


# %%
sns.pairplot(df)

# %%
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.offline as py
chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]
text1 = df["artist"] + " - " + df["name"]
text2 = text1.values

# X = data_frame.drop(droppable, axis=1).values
X = df[chosen].values
y = df["danceability"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

pca = PCA(n_components=3)
pca.fit(X)

X = pca.transform(X)

py.init_notebook_mode(connected=True)

trace = go.Scatter3d(x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    text=text2,
                    mode="markers",
                    marker=dict(size=8,color=y)
                    )

fig = go.Figure(data=[trace])
py.iplot(fig, filename="test-graph")

# %%
chosen = ["energy", "liveness", "tempo", "valence"]
text1 = df["artist"] + " - " + df["name"]
text2 = text1.values

# X = data_frame.drop(droppable, axis=1).values
X = df[chosen].values
y = df["loudness"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)

fig = {
    "data": [
        {
            "x": X[:, 0],
            "y": X[:, 1],
            "text": text2,
            "mode": "markers",
            "marker": {"size": 8, "color": y}
        }
    ],
    "layout": {
        "xaxis": {"title": "How danceable is this?"},
        "yaxis": {"title": "How classical is this?"}
    }
}

py.iplot(fig, filename="test-graph2")


# %%
import time

chosen = ["energy", "liveness", "tempo", "valence", "loudness",
          "speechiness", "acousticness", "danceability", "instrumentalness"]

X = df[chosen].values
y = df["loudness"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)


fig = {
    "data": [
        {
            "x": tsne_results[:, 0],
            "y": tsne_results[:, 1],
            "text": text2,
            "mode": "markers",
            "marker": {"size": 8, "color": y}
        }
    ],
    "layout": {
        "xaxis": {"title": "x-tsne"},
        "yaxis": {"title": "y-tsne"}
    }
}

py.iplot(fig, filename="test-graph2")


