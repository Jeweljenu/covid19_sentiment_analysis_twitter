import tkinter as tk
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from tkinter import ttk,messagebox
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from typing import List

pd.set_option("display.max_colwidth", 200) 
pd.set_option('display.max_colwidth', None)
plt.rcParams['figure.dpi'] = 115
plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

def load():
    loading_window = tk.Tk()
    loading_window.title("Loading...")
    loading_window.geometry("400x100")
    loading_window.state("zoomed")
    progress_label = tk.Label(loading_window, text="Loading...\nPlease Wait, This might take a minute....", font=("Helvetica", 16))
    progress_label.grid(row = 0,column = 0)
    
    progress_bar = ttk.Progressbar(loading_window, orient = "horizontal",length=300, mode="determinate")
    progress_bar.grid(row = 1,column = 0)
    loading_window.grid_rowconfigure(0,weight = 1)
    loading_window.grid_rowconfigure(1,weight = 1)
    loading_window.grid_columnconfigure(0,weight = 1)
    progress_bar.start()
    loading_window.update()
    progress_bar["value"] = 10
    
    working_dir_path = 'tweets_dataset.csv'
    df = pd.read_csv(working_dir_path,encoding = 'ISO-8859-1')
    loading_window.update()
    progress_bar["value"] = 20
    
    #print(df.head(30))
    df = df[['Tweet Posted Time (UTC)','Tweet Content','Tweet Location']]
    df.rename(columns={'Tweet Content': 'original tweets'}, inplace=True)
    #print(df.head())
    
    # write function for removing @user
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i,'',input_txt)
        return input_txt
    # create new column with removed @user
    df['Tweet'] = np.vectorize(remove_pattern)(df['original tweets'], '@[\w]*')
    loading_window.update()
    progress_bar["value"] = 25
    # remove special characters, numbers, punctuations
    df['Tweet'] = df['Tweet'].apply(lambda x: re.sub('[^a-zA-Z#]+', ' ', x))
    df.head(5)
    
    # function to get the subjectivity
    def get_subjectivity(Tweet: str) -> float:
        return TextBlob(Tweet).sentiment.subjectivity

    # function to get the polarity
    def get_polarity(Tweet: str) -> float:
        return TextBlob(Tweet).sentiment.polarity
    loading_window.update()
    progress_bar["value"] = 27
    # add new columns to dataframe - subjectivity, polarity
    df["subjectivity"] = df["Tweet"].apply(get_subjectivity)
    loading_window.update()
    progress_bar["value"] = 30
    df["polarity"] = df["Tweet"].apply(get_polarity)
    loading_window.update()
    progress_bar["value"] = 35
    
    #df.head()

    # function that computes the negative, neutral and positive sentiment
    def get_sentiment(score: float) -> str:
        if score < 0:
            return "negative"
        elif score == 0:
            return "neutral"
        else:
            return "positive"
        
    df["sentiment"] = df["polarity"].apply(get_sentiment)
    
    loading_window.update()
    progress_bar["value"] = 40
    
    #print(df.head())
    
    new_df = df[['Tweet','sentiment']]
    new_df.head()

    train,valid = train_test_split(new_df,test_size = 0.2,random_state=0,stratify = new_df.sentiment.values)
    #stratification means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.
    #print("train shape : ", train.shape)
    #print("valid shape : ", valid.shape)
    loading_window.update()
    progress_bar["value"] = 50
    
    stop = list(stopwords.words('english'))
    vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

    X_train = vectorizer.fit_transform(train.Tweet.values)
    X_valid = vectorizer.transform(valid.Tweet.values)

    y_train = train.sentiment.values
    y_valid = valid.sentiment.values
    loading_window.update()
    progress_bar["value"] = 60
    
    #print("X_train.shape : ", X_train.shape)
    #print("X_train.shape : ", X_valid.shape)
    #print("y_train.shape : ", y_train.shape)
    #print("y_valid.shape : ", y_valid.shape)
    
    # Create an instance of Logistic Regression classifier
    logistic_reg_clf = LogisticRegression(max_iter=1000)
    
    # Fit the classifier on the training data
    logistic_reg_clf.fit(X_train, y_train)

    # Make predictions on the validation data
    logistic_reg_prediction = logistic_reg_clf.predict(X_valid)

    # Calculate accuracy
    logistic_reg_accuracy = accuracy_score(y_valid, logistic_reg_prediction)
    
    ##print accuracy and classification report
    #print("Logistic Regression:")
    #print("Training accuracy Score:", logistic_reg_clf.score(X_train, y_train))
    #print("Validation accuracy Score:", logistic_reg_accuracy)
    #print(classification_report(logistic_reg_prediction, y_valid))
    loading_window.update()
    progress_bar["value"] = 70

    def mostloc(event):
        Top_Location_Of_tweet= df['Tweet Location'].value_counts().head(10)

        sns.set(rc={'figure.figsize':(11,7)})
        sns.set_style('white')
        Top_Location_Of_tweet_df=pd.DataFrame(Top_Location_Of_tweet)
        Top_Location_Of_tweet_df.reset_index(inplace=True)
        Top_Location_Of_tweet_df.rename(columns={'index':'Location', 'Tweet Location':'Location'}, inplace=True)
        viz_1=sns.barplot(x="Location", y="count", data=Top_Location_Of_tweet_df,palette='Blues_d')
        viz_1.set_title('Locations with most of the tweets')
        viz_1.set_ylabel('Count of listings')
        viz_1.set_xlabel('Location Names')
        viz_1.set_xticklabels(viz_1.get_xticklabels(),rotation=0)
        plt.show()
        #plt.savefig("mostloc.png")
        
    def get_percantage(df: pd.DataFrame, col: str) -> pd.DataFrame:
        data = df.get(col).value_counts().to_frame().rename(columns={"sentiment":"sentiment_total"})
        data["percentage"] = round(df.get(col).value_counts(normalize=True) * 100, 2)
        return data

    def COVID(event):
        def make_pie(df: pd.DataFrame, colors: List, header: str) -> None:
        
            # sort the index, because of the colors order - positive:green, neutral:blue, negative:red
            df = df.sort_index(ascending=False)
            total = sum(df.iloc[:,0])
            fig, ax = plt.subplots()
            ax.axis('equal')
            ax.set_prop_cycle("color", colors)
            outside, _ = ax.pie(df.iloc[:,0], radius=1.2, startangle=180)
            plt.setp(outside, width=0.3, edgecolor='white')
            ax.text(0, 0, f"{total}\ntweets", ha='center', va='center', size=25)
            ax.legend(['{:.0f}%: {}'.format(int(row.values) / total * 100, index) for index, row in df.iterrows()],frameon=False, bbox_to_anchor=(0.75, 0.02), labelspacing=0.7)
            ax.annotate(header, size=17, fontweight="semibold", xy=(1, 1), xycoords='data',horizontalalignment='center', verticalalignment='top', xytext=(0, 1.5))
            plt.show()
            #plt.savefig("covid.jpg")

        data = get_percantage(df, "sentiment")[['count']]
        colors = ['#5a7c38','#5080b0','#F2543D']
        header = 'Sentiment Analysis for keyword COVID'
        make_pie(data, colors=colors,header=header)

    loading_window.update()
    progress_bar["value"] = 80
    
    def scat(event):
        # display polarity and subjectivity
        def create_scatterplot(data: pd.DataFrame, title: str = "") -> None:
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(data["polarity"], data["subjectivity"], color="Blue")
            plt.title(f"COVID - Sentiment Analysis {title}", size=16)
            plt.xlabel("Polarity", size=12)
            plt.ylabel("Subjectivity", size=12)
            plt.show()
            #plt.savefig("scat.jpg")
        create_scatterplot(df)

    def wcloud(event):
        # create text from all tweets
        all_words = ' '.join([text for text in df['Tweet']])

        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()
        #plt.savefig("wcloud.jpg")

    def top10(event):
        # function to collect hashtags
        def hashtag_extract(x):
            hashtags = []
            for i in x:
                ht = re.findall(r'#(\w+)', i)
                hashtags.append(ht)
            return hashtags
        # extracting hashtags from non racist/sexist tweets
        HT_Positive = hashtag_extract(df['original tweets'][df['sentiment'] == 'positive'])
        HT_Neutral = hashtag_extract(df['original tweets'][df['sentiment'] == 'neutral'])
        HT_Negative = hashtag_extract(df['original tweets'][df['sentiment'] == 'negative'])
        # unnesting list
        HT_Positive = sum(HT_Positive, [])
        HT_Neutral = sum(HT_Neutral, [])
        HT_Negative = sum(HT_Negative,[])
        # making frequency distribution top 10 Positive hashtags
        a = nltk.FreqDist(HT_Positive)
        d = pd.DataFrame({'Hashtag': list(a.keys()),'Count' : list(a.values())})

        d = d.nlargest(columns = 'Count', n = 10)

        plt.figure(figsize = (16,5))
        ax = sns.barplot(data =d, x = 'Hashtag', y = 'Count')
        plt.show()
        #plt.savefig("top10.jpg")

    def senti(event):
        global textinput,root,frame1
        custom = textinput.get()
        if custom == "":
            messagebox.showerror("Input Error","Enter the Tweet!!!")
        else:
            custom_test_data = [custom]
            X_custom_test = vectorizer.transform(custom_test_data)# Use the model to make predictions on the custom test data
            custom_predictions = logistic_reg_clf.predict(X_custom_test)
            #print("predicted sentiment: "+custom_predictions[0])
            output = tk.Label(frame1,text= custom_predictions[0].upper(),font = ("Algerian",20))
            output.grid(row = 1,column = 3)
        
    def back2(event):
        global root2
        root2.withdraw()
        home('x')
        
    def display(event):
        global root2,dropinput
        s = dropinput.get()
        if(s == "Location with most of the Tweets"):
            mostloc('')
        elif (s == "Top 10 Most #Hashtags"):
            top10('')
        elif (s == "Sentiment Analysis for keyword COVID"):
            COVID('')
        elif (s == "COVID - Sentiment Analysis"):
            scat('')
        else:
            wcloud('')
                
    def analy(event):
        global root1,root2,dropinput
        root1.withdraw()
        root2 = tk.Tk()
        root2.state("zoomed")
        root2.title("Sentiment Analysis - Model Analysis")
        root2.geometry("700x350")
        root2['bg'] = 'white'
        frame2 = tk.Frame(root2,highlightbackground = "black",highlightthickness = 2)
        frame2.grid(row = 1,column = 1,padx = 150,pady = 150)
        text = tk.Label(frame2,text = "Enter your Choice:",font = ("Lucida Calligraphy",20),fg = "black")
        text.grid(row = 0,column = 2,padx = 150,pady = 150)
        dropinput = ttk.Combobox(frame2,values = ["Location with most of the Tweets","Top 10 Most #Hashtags","Sentiment Analysis for keyword COVID","COVID - Sentiment Analysis","Word Cloud of the most words"],font = ('Times New Roman',20),state = 'readonly')
        dropinput.grid(row = 0,column = 4,padx = 150)
        dropinput.current(0)
        submit = tk.Button(frame2,text = "Submit",font = ("Lucida Calligraphy",20),fg = "black",bg = "lime")
        submit.grid(row = 2,column = 2)
        submit.bind('<Button-1>',display)
        back = tk.Button(frame2,text = "Back",font = ("Lucida Calligraphy",20),fg = "black",bg = "grey")
        back.grid(row = 2,column = 4)
        back.bind('<Button-1>',back2)
        
    def back1(event):
        global root
        root.withdraw()
        home('X')

    def home(event):
        global root1
        root1 = tk.Tk()
        root1.state('zoomed')
        root1.title("Sentiment Analysis")
        root1['bg'] = "white"
        title1 = tk.Label(root1,fg = "blue",bg = "white",text = "Sentiment Analysis of Lockdown\nin India during COVID-19",font = ('Algerian',30))
        title1.grid(row = 0,column= 2,pady = 50,padx = 50)
        pred = tk.Button(root1,text = "Sentiment\nPrediction",bg = "white",width = 10,height = 2,font = ('Lucida Calligraphy',25))
        pred.grid(row = 1,column = 1,padx = 75)
        pred.bind('<Button-1>',main)
        graph1 = tk.Button(root1,text = "Model\nAnalysis",bg = "white",width = 10,height = 2,font = ('Lucida Calligraphy',25))
        graph1.grid(row= 1,column = 3)
        graph1.bind('<Button-1>',analy)
        tk.Button(root1, text="Quit",fg = 'white',bg = '#E41B17',command=root1.destroy,width = 5,height = 2,font = ('Lucida Calligraphy',25)).grid(row=2,column=2,pady=200)

    def main(event):
        global textinput,root,root1,frame1
        root1.withdraw()
        root = tk.Tk()
        root.state('zoomed')
        root.title("Sentiment Analysis - Prediction")
        root.geometry("700x350")
        root['bg'] = 'white'
        frame1 = tk.Frame(root,highlightbackground = "black",highlightthickness = 2)
        frame1.grid(row = 1,column = 1,padx = 150,pady = 150)
        text = tk.Label(frame1,text = "Enter your tweet:",font = ("Lucida Calligraphy",20),fg = "black")
        text.grid(row = 0,column = 2,padx = 150,pady = 150)
        textinput = tk.Entry(frame1,font = ('Times New Roman',20))
        textinput.grid(row = 0,column = 4,padx = 150)
        submit = tk.Button(frame1,text = "Submit",font = ("Lucida Calligraphy",20),fg = "black",bg = "lime")
        submit.grid(row = 2,column = 2)
        submit.bind('<Button-1>',senti)
        back = tk.Button(frame1,text = "Back",font = ("Lucida Calligraphy",20),fg = "black",bg = "grey")
        back.grid(row = 2,column = 4)
        back.bind('<Button-1>',back1)
        output = tk.Label(frame1,font = ("Algerian",20))
        output.grid(row = 1,column = 3)
    loading_window.update()
    progress_bar["value"] = 90
    loading_window.update()
    progress_bar["value"] = 100
    progress_bar.stop()
    loading_window.destroy()       
    home('')

load()
