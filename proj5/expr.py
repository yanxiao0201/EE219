from common import *

util.tic('Loading tweet data...')
tweets = load_tweets(['#gopatriots'])
tweet = tweets['#gopatriots'][0]
util.toc()

"""

In [18]: tweet.keys()
Out[18]: 
[u'firstpost_date',
 u'author',
 u'url',
 u'tweet',
 u'title',
 u'original_author',
 u'citation_date',
 u'metrics',
 u'highlight',
 u'type',
 u'citation_url']

In [23]: tweet['firstpost_date']
Out[23]: 1420835445

In [24]: time.time()
Out[24]: 1489215925.12033

In [27]: tweet['tweet'].keys()
Out[27]: 
[u'contributors',
 u'truncated',
 u'text',
 u'in_reply_to_status_id',
 u'id',
 u'favorite_count',
 u'source',
 u'retweeted',
 u'coordinates',
 u'timestamp_ms',
 u'entities',
 u'in_reply_to_screen_name',
 u'in_reply_to_user_id',
 u'retweet_count',
 u'id_str',
 u'favorited',
 u'user',
 u'geo',
 u'in_reply_to_user_id_str',
 u'possibly_sensitive',
 u'lang',
 u'created_at',
 u'filter_level',
 u'in_reply_to_status_id_str',
 u'place']


In [11]: tweet['metrics']
Out[11]: 
{u'acceleration': 0,
 u'citations': {u'data': [{u'citations': 0, u'timestamp': 1421612999}],
  u'influential': 0,
  u'matching': 1,
  u'replies': 0,
  u'total': 1},
 u'impressions': 1330,
 u'momentum': 0,
 u'peak': 0,
 u'ranking_score': 4.412375}


In [16]: tweet['author']
Out[16]: 
{u'author_img': u'http://pbs.twimg.com/profile_images/562109096519012352/49vxe_9q_normal.jpeg',
 u'description': u"I'm a die-hard Patriots, Suns, Badgers and Brewers fan #TeamPatriots | #PatsNation | #FuelTheFire | #GoSuns | #Brewcrew | #Bucks | #FeerTheDeer",
 u'followers': 2895.0,
 u'image_url': u'http://pbs.twimg.com/profile_images/562109096519012352/49vxe_9q_normal.jpeg',
 u'name': u'Alex Kroll',
 u'nick': u'patsnation87',
 u'type': u'twitter',
 u'url': u'http://twitter.com/patsnation87'}


In [17]: tweet['tweet']['entities']
Out[17]: 
{u'hashtags': [{u'indices': [43, 61], u'text': u'ThrowbackThursday'},
  {u'indices': [129, 140], u'text': u'GoPatriots'}],
 u'symbols': [],
 u'trends': [],
 u'urls': [{u'display_url': u'vine.co/v/OphHATaAggX',
   u'expanded_url': u'https://vine.co/v/OphHATaAggX',
   u'indices': [62, 85],
   u'url': u'https://t.co/F5FX5KVmdX'}],
 u'user_mentions': []}

"""

