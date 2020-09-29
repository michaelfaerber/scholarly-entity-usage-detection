import praw
from pymongo import MongoClient
import datetime
client = MongoClient('localhost:27017')
db = client.reddit

def store_redditpost_in_mongo(db, id, title, created, submission,subreddit):
    my_ner = {
        "post_id": id,
        "title": title,
        "created":created,
        "submission":submission,
        "subreddit":subreddit
    }
    db.reddit_post.insert_one(my_ner)
    return True


def store_redditcomment_in_mongo(db, id, comment):
    my_ner = {
        "post_id": id,
        "comment": comment
    }

    db.reddit_comment.insert_one(my_ner)
    return True

reddit = praw.Reddit(client_id='nI7zQphvoWq2hg',
                     client_secret='7hRVTJE2yZ_hGs5qXmJJ-SM_WWo',
                     user_agent='my user agent')

subreddits = ['AskDocs', 'medicalschool', 'medicine', 'DiagnoseMe']      # 'doctorwho'

for sub in subreddits:
    print('Retrieving:', sub)
    for submission in reddit.subreddit(str(sub)).hot():
        print(submission.id)
        try:
            store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext, str(sub))
            for top_level_comment in submission.comments:
                store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
        except:
            pass
    print('-' * 100)
    print('')

# for submission in reddit.subreddit('AskDocs').hot():
#     # comment=reddit.submission(submission)
#     # print(submission.title)
#     # print(datetime.datetime.fromtimestamp(submission.created))
#     print(submission.id)
#     # print(submission.selftext)
#     try:
#         store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext, 'AskDocs')
#         for top_level_comment in submission.comments:
#             store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
#     except:
#         pass
        
# for submission in reddit.subreddit('medicalschool').hot():
#     try:
#         store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext, 'medicalschool')
#         for top_level_comment in submission.comments:
#             store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
#     except:
#         pass

# for submission in reddit.subreddit('doctorwho').hot():
#     # comment=reddit.submission(submission)
#     try:
#         store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext,'doctorwho')
#         for top_level_comment in submission.comments:
#             store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
#     except:
#         pass

# for submission in reddit.subreddit('medicine').hot():
#     # comment=reddit.submission(submission)
#     try:
#         store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext, 'medicine')
#         for top_level_comment in submission.comments:
#             store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
#     except:
#         pass
    
# for submission in reddit.subreddit('DiagnoseMe').hot():
#     # comment=reddit.submission(submission)
#     try:
#         store_redditpost_in_mongo(db, submission.id, submission.title, datetime.datetime.fromtimestamp(submission.created), submission.selftext, 'DiagnoseMe')
#         for top_level_comment in submission.comments:
#             store_redditcomment_in_mongo(db, submission.id, top_level_comment.body)
#     except:
#         pass

# # for moderator in reddit.subreddit('AskDocs').moderator():
# #     print(moderator)