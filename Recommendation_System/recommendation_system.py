

# Commented out IPython magic to ensure Python compatibility.
# Importing Libraries
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
warnings.filterwarnings('ignore')

"""### Load and Check Data"""

books = pd.read_csv('C:/Users/Masum/Downloads/book.csv', encoding='Latin1')
books



print('Number of Unique Users are {}'.format(len(books['user_id'].unique())))

print('Number of Unique Books are {}'.format(len(books['title'].unique())))

books['rating'].value_counts()

books['user_id'].unique()

books.info()

books.rating.describe()

books.describe()

books.isnull().any()

books.isnull().sum()

books.duplicated().sum()

books[books.duplicated()].shape

books[books.duplicated()]

"""Let's create a ratings dataframe with average rating and number of ratings:"""

books.groupby('title')['rating'].mean().sort_values(ascending=False).head()


books.groupby('title')['rating'].count().sort_values(ascending=False).head(10)

ratings = pd.DataFrame(books.groupby('title')['rating'].mean())
ratings.head()

ratings['num of ratings'] = pd.DataFrame(books.groupby('title')['rating'].count())
ratings.head()

"""### Data Visualization<a class="anchor" id="3"></a>"""

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

"""##### Observation: Maximum Number of Books are Rated only Once"""

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

plt.figure(figsize=(10,6))
books['rating'].value_counts().plot(kind='bar')
plt.title('Ratings Frequency',  fontsize = 18, fontweight = 'bold')

"""### Observations:
+ #### Most Frequent Ratings by Users are: 8,7 and 10
"""

top_books = books['title'].value_counts().head(10)
top_books.index

plt.figure(figsize = (16,9))
plt.pie(top_books,
       labels=top_books.index,
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Most Frequent Books Bought", fontsize = 18, fontweight = 'bold')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(books.title))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()

"""### Data Pre-Processing<a """

# Renaming the columns name
books.rename({'Unnamed: 0':'index','User.ID':'user_id','Book.Title':'title','Book.Rating':'rating'}, axis= 1, inplace =True)
books.set_index('index', inplace=True)
books



user_books_df = books.pivot_table(index='user_id',columns = 'title', values = 'rating').fillna(0)
user_books_df

# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation, jaccard

ratings.sort_values('num of ratings',ascending=False).head(10)

stardust_user_ratings = user_books_df['Stardust']
fahrenheit_user_rating = user_books_df['Fahrenheit 451']
fahrenheit_user_rating.head()

"""We can then use corrwith() method to get correlations between two pandas series:"""

similar_to_fahrenheit = user_books_df.corrwith(fahrenheit_user_rating)
similar_to_stardust = user_books_df.corrwith(stardust_user_ratings)

corr_fahrenheit = pd.DataFrame(similar_to_fahrenheit,columns=['Correlation'])
corr_fahrenheit.dropna(inplace=True)
corr_fahrenheit.head()



corr_fahrenheit.sort_values('Correlation',ascending=False).head(10)



corr_fahrenheit = corr_fahrenheit.join(ratings['rating'])
corr_fahrenheit.head()


corr_fahrenheit[corr_fahrenheit['rating']>5].sort_values('Correlation',ascending=False).head()

corr_stardust = pd.DataFrame(similar_to_stardust,columns=['Correlation'])
corr_stardust.dropna(inplace=True)
corr_stardust = corr_stardust.join(ratings['num of ratings'])
corr_stardust[corr_stardust['num of ratings']>4].sort_values('Correlation',ascending=False).head()



user_books_df.head()

# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation, jaccard



user_sim = 1 - pairwise_distances(user_books_df.values, metric = 'cosine')
user_sim

# Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df

user_sim_df.iloc[:5,:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5,0:5]

# Set the index and column name to user Ids
user_sim_df.index = list(user_books_df.index)
user_sim_df.columns = list(user_books_df.index)
user_sim_df

user_id_eight = user_sim_df.sort_values([9], ascending=False).head(100)
user_id_eight[9]

books[(books['user_id']==8) | (books['user_id']==14)]

# Most Similar Users
user_sim_df.idxmax(axis=1)

books[(books['user_id']==8) | (books['user_id']==14)]



def give_reco(customer_id):
    tem = list(user_sim_df.sort_values([customer_id],ascending=False).head(100).index)
    #print('similar customer ids:',tem)
    movie_list=[]
    for i in tem:
        movie_list=movie_list+list(books[books['user_id']==i]['title'])
    #print('Common movies within customer',movie_list)
    return set(movie_list)-set(books[books['user_id']==customer_id]['title'])

give_reco(14)

give_reco(8)

