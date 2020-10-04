from google.cloud import pubsub_v1
import base64
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json

PROJECT_ID = "engineering123"
TOPIC = "twitter_subject"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC)

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

def publish(publisher, topic_path, data_lines):
    messages = []
    for line in data_lines:
        messages.append({'data': line})
    body = {'messages': messages}
    str_body = json.dumps(body)
    data = base64.urlsafe_b64encode(bytearray(str_body, 'utf8'))
    print(data)
    return publisher.publish(topic_path, data=data)

def callback(message_future):
    # When timeout is unspecified, the exception method waits indefinitely.

    if message_future.exception(timeout=30):
        print('Publishing message on twitter_subject threw an Exception {}.'.format(
            message_future.exception()))
    else:
        print(message_future.result())

class StdOutListener(StreamListener):
    """A listener handles tweets that are received from the stream.
    This listener dumps the tweets into a PubSub topic
    """
    def __init__(self):
        super(StreamListener, self).__init__()
        self.tweets = []

    def on_status(self, status):
        """What to do when tweet data is received."""
        if status.user.location is not None:
            date = status.created_at
            str_date = date.strftime("%Y-%m-%d %H:%M:%S")
            text = status.text
            user_id = status.user.screen_name
            loc = status.user.location
            print(text)
            print(user_id)
            print(loc)
            tw = dict(date=str_date, text=text, user_id = user_id, location=loc)
            print(tw)
            self.tweets.append(tw)
            publish(publisher, topic_path, self.tweets)
            self.tweets = []

        else:
            pass
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    print('시작')
    listener = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener)
    stream.filter(
                track=['trump', 'biden']
                )