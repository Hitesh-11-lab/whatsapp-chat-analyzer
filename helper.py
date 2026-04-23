from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob
import string

extract = URLExtract()

# ==============================
# 🔥 CUSTOM ABUSIVE WORD LIST
# ==============================

ABUSIVE_WORDS = [

    # ENGLISH
    "idiot","stupid","dumb","fool","moron","loser",
    "useless","nonsense","trash","garbage","jerk",
    "hate","damn","hell","bastard","bloody","crap",
    "retard","retarded","psycho","crazy",
    "pathetic","disgusting","cheap","ugly",
    "worthless","liar","fake","fraud","scam",
    "noob","clown",

    # HINGLISH / HINDI
    "pagal","gadha","bewakoof","bekaar",
    "bakwas","faltu","chutiya","chutia",
    "kutte","kutta","harami","kamina",
    "saala","saali","ullu","nalayak",
    "jhatu","jhant","lund","bhosdike",
    "bhenchod","behenchod","madarchod",
    "mc","bc","bsdk","randi","raand",
    "chodu","gandu","gaand","lavde",
    "lavda","lodu","tatti","kuttiya",

    # Variations
    "chutya","chutiye","haramkhor",
    "kamini","gand","ganduu"
]

# ==============================
# 📊 BASIC STATS
# ==============================

def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(str(message).split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(str(message)))

    return num_messages, len(words), num_media_messages, len(links)


# ==============================
# 👥 MOST BUSY USERS
# ==============================

def most_busy_users(df):
    x = df['user'].value_counts().head()

    percent_df = round(
        (df['user'].value_counts() / df.shape[0]) * 100, 2
    ).reset_index().rename(columns={'index': 'name', 'user': 'percent'})

    return x, percent_df


# ==============================
# ☁ WORDCLOUD
# ==============================

def create_wordcloud(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != '<Media omitted>\n'].copy()

    def remove_stop_words(message):
        words = []
        for word in str(message).lower().split():
            if word not in stop_words:
                words.append(word)
        return " ".join(words)

    temp['clean_message'] = temp['message'].apply(remove_stop_words)

    wc = WordCloud(width=500, height=500,
                   min_font_size=10,
                   background_color='white')

    df_wc = wc.generate(temp['clean_message'].str.cat(sep=" "))
    return df_wc


# ==============================
# 🔥 MOST COMMON WORDS
# ==============================

def most_common_words(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in str(message).lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))


# ==============================
# 😂 EMOJI ANALYSIS
# ==============================

def emoji_helper(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []

    for message in df['message']:
        emojis.extend([c for c in str(message) if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))


# ==============================
# 📅 TIMELINES
# ==============================

def monthly_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline


def daily_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


def week_activity_map(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)


# ==============================
# 😊 SENTIMENT ANALYSIS
# ==============================

def sentiment_analysis(selected_user, df):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    df = df.copy()

    sentiments = []

    for message in df['message']:
        polarity = TextBlob(str(message)).sentiment.polarity

        if polarity > 0:
            sentiments.append("Positive")
        elif polarity < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    df['sentiment'] = sentiments

    return df['sentiment'].value_counts(), df


# ==============================
# ⏱ REPLY TIME ANALYSIS
# ==============================

def reply_time_analysis(selected_user, df):

    df = df.copy().sort_values('date')

    df['reply_time'] = df['date'].diff().dt.total_seconds()
    df = df.dropna(subset=['reply_time'])

    df = df[df['user'] != df['user'].shift(1)]
    df['reply_time_minutes'] = df['reply_time'] / 60

    df = df[(df['reply_time_minutes'] > 0) & (df['reply_time_minutes'] < 360)]

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    return df.groupby('user')['reply_time_minutes'].median().sort_values()


# ==============================
# 🔥 ABUSE ANALYSIS
# ==============================

def abuse_analysis(selected_user, df):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    df = df.copy()

    abusive_words_used = []
    abusive_user_list = []
    word_user_map = {}

    for _, row in df.iterrows():

        message = str(row['message']).lower()
        message = message.translate(str.maketrans('', '', string.punctuation))
        words = message.split()

        for word in words:
            if word in ABUSIVE_WORDS:

                abusive_words_used.append(word)
                abusive_user_list.append(row['user'])

                if word not in word_user_map:
                    word_user_map[word] = {}

                word_user_map[word][row['user']] = \
                    word_user_map[word].get(row['user'], 0) + 1

    total_abusive = len(abusive_words_used)

    abuse_percentage = (
        (total_abusive / len(df)) * 100
        if len(df) > 0 else 0
    )

    abuse_by_user = Counter(abusive_user_list)

    most_abusive_user = (
        max(abuse_by_user, key=abuse_by_user.get)
        if abuse_by_user else "No abuse found"
    )

    abusive_word_df = pd.DataFrame(
        Counter(abusive_words_used).most_common(10),
        columns=["Abusive Word", "Count"]
    )

    abuse_by_user_series = pd.Series(abuse_by_user)

    return (
        total_abusive,
        round(abuse_percentage, 2),
        most_abusive_user,
        abuse_by_user_series,
        abusive_word_df,
        word_user_map
    )


# ==============================
# 📄 USER-WISE ABUSE TABLE
# ==============================

def abusive_word_user_table(df):

    records = []

    for _, row in df.iterrows():

        user = row["user"]
        message = str(row["message"]).lower()
        message = message.translate(str.maketrans('', '', string.punctuation))
        words = message.split()

        for word in words:
            if word in ABUSIVE_WORDS:
                records.append((user, word))

    counter = Counter(records)

    data = []

    for (user, word), count in counter.items():
        data.append({
            "User": user,
            "Abusive Word": word,
            "Count": count
        })

    return pd.DataFrame(data)