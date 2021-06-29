from google.protobuf.symbol_database import Default
import nltk
import random
import pickle
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from numpy.lib.function_base import append
lemmatizer = WordNetLemmatizer()
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import time
import plotly.graph_objects as go
import requests
import json
import numpy as np
from keras.models import load_model
from bs4 import BeautifulSoup
import csv

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model = load_model("stock_model.h5")
intents=json.loads(open('training.json').read())
def calcMovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['Adj Close'].rolling(size).mean()
    df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
    df.dropna(inplace=True)
    return df

def calc_macd(data):
    df = data.copy()
    df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
    df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df.dropna(inplace=True)
    return df

def calcBollinger(data, size):
    df = data.copy()
    df["sma"] = df['Adj Close'].rolling(size).mean()
    df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
    df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
    df["width"] = df["bolu"] - df["bold"]
    df.dropna(inplace=True)
    return df


def graphMyStock(finalvar,a,b,col):
    stock2 = yf.Ticker(finalvar)
    info2=stock2.info
    ln2=info2['longName']
            
    opt1b, opt2b = st.beta_columns(2)
    with opt1b:
        numYearMAb = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=a)    
            
    with opt2b:
            windowSizeMAb = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=b) 
    start2 = dt.datetime.today()-dt.timedelta(numYearMAb * 365)
    end2 = dt.datetime.today()
    livedata2 = yf.download(finalvar,start2,end2)
    df_ma2 = calcMovingAverage(livedata2, windowSizeMAb)
    df_ma2 = df_ma2.reset_index()
                        
    fig2 = go.Figure()
                    
    fig2.add_trace(
        go.Scatter(
                x = df_ma2['Date'],
                y = df_ma2['Adj Close'],
                name = '('+ finalvar+ ') '+ "Prices Over Last " + str(numYearMAb) + " Year(s)",
                mode='lines',
                line=dict(color=col)
                                )
                        )          
    fig2.update_layout(showlegend=True,legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
                ))
                        
    fig2.update_layout(legend_title_text='Trend')
    fig2.update_yaxes(tickprefix="$")
    st.plotly_chart(fig2, use_container_width=True) 

def graphAllStocks(stocka,stockb,stockc,a,b,col1,col2,col3):
    stock2 = yf.Ticker(stocka)
    info2=stock2.info
    ln2=info2['longName']
    st.write('')
    st.subheader('**Graph of optimal stocks:** ')

            
    opt1b, opt2b = st.beta_columns(2)
    with opt1b:
        numYearMAb = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=a)    
            
    with opt2b:
            windowSizeMAb = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=b) 
    start2 = dt.datetime.today()-dt.timedelta(numYearMAb * 365)
    end2 = dt.datetime.today()
    livedata2 = yf.download(stocka,start2,end2)
    df_ma2 = calcMovingAverage(livedata2, windowSizeMAb)
    df_ma2 = df_ma2.reset_index()
                        
    fig2 = go.Figure()
                    
    fig2.add_trace(
        go.Scatter(
                x = df_ma2['Date'],
                y = df_ma2['Adj Close'],
                name = '('+ stocka+ ') '+ "Prices Over Last " + str(numYearMAb) + " Year(s)",
                mode='lines',
                line=dict(color=col1)
                                )
                        )   
    livedata2=yf.download(stockb,start2,end2)
    df_ma2= calcMovingAverage(livedata2, windowSizeMAb)
    df_ma2= df_ma2.reset_index()
    fig2.add_trace(
        go.Scatter(
                x=df_ma2['Date'],
                y=df_ma2['Adj Close'],
                name = '('+ stockb+ ') '+ "Prices Over Last " + str(numYearMAb) + " Year(s)",
                mode='lines',
                line=dict(color=col2)
                    )) 
    
    livedata3=yf.download(stockc,start2,end2)
    df_ma3= calcMovingAverage(livedata3, windowSizeMAb)
    df_ma3= df_ma3.reset_index()
    fig2.add_trace(
        go.Scatter(
                x=df_ma3['Date'],
                y=df_ma3['Adj Close'],
                name = '('+ stockc+ ') '+ "Prices Over Last " + str(numYearMAb) + " Year(s)",
                mode='lines',
                line=dict(color=col3)
                    ))         

    fig2.update_layout(showlegend=True,legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
                ))
                        
    fig2.update_layout(legend_title_text='Trend')
    fig2.update_yaxes(tickprefix="$")
    st.plotly_chart(fig2, use_container_width=True) 

def RootWordGen(lw):
    j=nltk.word_tokenize(lw)
    j= [lemmatizer.lemmatize(word.lower()) for word in j]

    return(j)


def matrix(sentence, words, show_details=True):
    sentence_words= RootWordGen(sentence)
    # sentence_words is bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    #matrix contains number of elements = vocabulary, preset value=0
    for s in sentence_words:
        #traverses root words
        for i,w in enumerate(words):
            #i is roll no/dir no
            #w is unique word
            #makes directory, gives a 'roll no' to each word. If 'cramping' is entered, directory till cramping prints along w roll number, then matrix with 0s other than one 1 (one being element number=roll no of cramping)
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    #will give name of bag of unique base word the entered word is found in
                    print ("found in bag: %s" % w) 
    #removes commas from list, returns matrix
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold probability
    pred= matrix(sentence, words,show_details=False)
    res = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25
    global results
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    global results1
    results1 = [[i,r] for i,r in enumerate(res)]
    print(results)

    #for guesses above threshold
    f=open('r.txt','w')
    #for all guesses
    f1=open('s.txt','w')
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    results1.sort(key=lambda x: x[1], reverse=True)
    pr=results1[0]
    global pp
    pp=pr[1]
    print(pp)
    global return_list
    return_list = []
    global return_list1
    return_list1=[]
    for r in results1:
        return_list1.append({"intent": classes[r[0]], "probability": str(r[1])})
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    for x in return_list1:
        f1.write(str(x))
    for x in return_list:
        print(x)
        f.write(str(x))
    return return_list[0]

def getResponse(ints, intents_json):
    global tag
    tag = ints[0]['intent']
    print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def FinalPrediction(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

stockdata = pd.read_csv("SP500.csv")
symbols = stockdata['Symbol'].sort_values().tolist()  

st.title('Investment Optimizer and Stock Growth Predictor')

#We'll add this when we come up with something
expander=st.beta_expander(label='',expanded=False)
expander.write("**This application aims at evaluating stock trends and current news to predict it's future growth. It provides a clean and efficient user interface to view current prices and fluctuation history. It also provides a tool to identify a ideal combination of stocks that one should invest in based on the given budget, using our machine learning and optimization algorithm. **")

st.write("")
st.write("")
st.write('**Would you like to know where to invest or understand each Stock?**')
a=st.radio("", ("Invest", "Understand"))

if(a=="Invest"):
    budget=st.sidebar.number_input("Enter your budget ($): ")
    if(st.sidebar.button("Enter")):
        st.header("")
        st.header("**Following is the combination of stocks you should invest in:  ** ")
        st.write("")
        st.write('Processing...')
        invest=[]
        invstock_sym=[]
        invstock_name=[]
        f= open("SP500.csv",'r')
        rd=csv.reader(f)
        for x in rd:
            if x!=[]:
                if x[2]=='badboy':
                    invstock_sym.append(x[0])
                    invstock_name.append(x[1])
        invstock_price=[]
        for ticker in invstock_sym:
            ticker_yahoo = yf.Ticker(ticker)
            data = ticker_yahoo.history()
            last_quote = (data.tail(1)['Close'].iloc[0])
            invstock_price.append(float(last_quote))    
        invstock_conf=[]
        st.markdown("""
            <style>
            .stProgress .st-bo {
                background-color: green;
            }
            </style>
            """, unsafe_allow_html=True)
        my_bar=st.progress(0)
        progresscount=10

        for badgirl in invstock_name:
            send="https://www.google.com/search?q=should+you+invest+in+ "+badgirl.lower()+" stock"
            res=requests.get(send)
            soup=BeautifulSoup(res.content, "html.parser")
            all_links=[]
            count=0
            for i in soup.select("a"):
                if count==1:
                    break
                link=i.get("href")
                if("/url?q=https://" in link):
                    if(("/url?q=https://support.google.com" not in link) and ("/url?q=https://accounts.google.com" not in link)):
                        x=link.split("https://")
                        y=x[1].split("&sa")
                        new="https://"+y[0]
                        all_links.append(new)
                        z=i.text
                        if("..." in z):
                            type2=z.split("...")
                            name=type2[0]
                        else:
                            type1=z.split(" › ")
                            name=type1[0]
                        count+=1
            list1=[]
            c=0
            for i in all_links:
                if c==1:
                    break
                option=requests.get(i)
                soup=BeautifulSoup(option.content, "html.parser")
                pageinfo=soup.select("p")
                for j in pageinfo:
                    m=j.text
                    n=m.split(' ')
                    for i in n:
                        list1.append(i)
                c=c+1   
            tex=' '.join(list1)   
            find=predict_class(tex,model)
            varun=[]
            varun.append(float(find['probability']))
            varun.append(find['intent'])
            invstock_conf.append(varun)
            progresscount=progresscount+10
            my_bar.progress(progresscount)
        stocks={}
        for i in range(len(invstock_name)):
            temp=[]
            if invstock_conf[i][1]=='up':
                temp.append(invstock_conf[i][0])
                temp.append(invstock_price[i])
                temp.append(invstock_name[i])
                temp.append(invstock_sym[i])
                length= len(stocks)
                stocks[length]=temp

        ###### NEED TO GET "STOCKS" DICTIONARY DATA FROM ######## 
        all_stocks={}
        st.write(str(stocks))
        for i in range(len(stocks)):
            if((budget >= stocks[i][1]) and (stocks[i][0]>0.5)):
                n=len(all_stocks)
                all_stocks[n]=[stocks[i][0], stocks[i][1], stocks[i][2], stocks[i][3]]
        if len(all_stocks)>=3:
            st.balloons()
            quad1={}
            quad2={}
            quad3={}
            quad4={}

            for i in range(len(all_stocks)):

                if((all_stocks[i][0]>=0.8) and (all_stocks[i][1]<=100)):
                    quad1[i]=[all_stocks[i][0], all_stocks[i][1], all_stocks[i][2],all_stocks[i][3]]
                elif((all_stocks[i][0]>=0.8) and (all_stocks[i][1]>100)):
                    quad2[i]=[all_stocks[i][0], all_stocks[i][1], all_stocks[i][2],all_stocks[i][3]]
                elif((all_stocks[i][0]<0.8) and (all_stocks[i][1]<=100)):
                    quad3[i]=[all_stocks[i][0], all_stocks[i][1], all_stocks[i][2],all_stocks[i][3]]
                else:
                    quad4[i]=[all_stocks[i][0], all_stocks[i][1], all_stocks[i][2],all_stocks[i][3]]

            def inputs(quad):
                global invest
                spq=[]
                for i in quad:
                    spq.append(quad[i][1])
                length=len(spq)
                for i in range(length):
                    if(len(invest)==3):
                        break
                    minval=min(spq)
                    for i in quad:
                        if(quad[i][1]==minval):
                            invest.append(quad[i])
                    spq.remove(minval)

            inputs(quad1)
            if(len(invest)<3):
                inputs(quad2)
            if(len(invest)<3):
                inputs(quad3)
            if(len(invest)<3):
                inputs(quad4)

            #stock1 should get 60%
            #stock2 should get 30%
            #stock3 should get 10%

            s1=budget*0.6
            s2=budget*0.3
            s3=budget*0.1

            n_s1=s1//invest[0][1]
            n_s2=s2//invest[1][1]
            n_s3=s3//invest[2][1]

            left=budget-invest[0][1]*n_s1-invest[1][1]*n_s2-invest[2][1]*n_s3

            invest_val=[]
            for i in range(3):
                invest_val.append(invest[i][1])

            a_s1=0
            a_s2=0
            a_s3=0

            a_s3=left//invest[2][1]
            left=left-a_s3*invest[2][1] 
            a_s2=left//invest[1][1]
            left=left-a_s2*invest[1][1]
            a_s1=left//invest[0][1]
            left=left-a_s1*invest[0][1]    

            t_s1=n_s1+a_s1
            t_s2=n_s2+a_s2
            t_s3=n_s3+a_s3
            
            st.write("")
            st.subheader('**Summary:** ')
            summary_table={}
            names=[]
            prices=[]
            nstocks=[]
            totalcosts=[]
            confidences=[]
            for i in range(len(invest)):
                names.append(invest[i][2])
                prices.append(invest[i][1])
                if(i==0):
                    nstocks.append(t_s1)
                    tcost=t_s1*invest[i][1]
                    totalcosts.append(tcost)
                if(i==1):
                    nstocks.append(t_s2)
                    tcost=t_s2*invest[i][1]
                    totalcosts.append(tcost)
                if(i==2):
                    nstocks.append(t_s3)
                    tcost=t_s3*invest[i][1]
                    totalcosts.append(tcost)
                confidences.append(invest[i][0])

            summary_table["Stock Name"]=names
            summary_table["Cost per Stock"]=prices
            summary_table["Number to Purchase"]=nstocks
            summary_table["Total Cost"]=totalcosts
            summary_table["Our Confidence"]=confidences

            column_order=["Stock Name", "Cost per Stock", "Number to Purchase", "Total Cost", "Our Confidence"]
            summary_df=pd.DataFrame(data=summary_table)
            st.dataframe(summary_df)
            st.write("")
            bala='**Your balance:** '+ '_$' + str(left) +'_'
            st.write(bala)
            graphAllStocks(invest[0][3],invest[1][3],invest[2][3],14,15,'royalblue','springgreen','indianred')

            st.header('**In depth review:** ')
            st.write('')
            text1='Your first stock: ' + '_' + str(invest[0][2]) + '_'
            st.header(text1)
            graphMyStock(invest[0][3],1,2,'royalblue') 
            
            text1a='**Price:** '+ '_$'+ str(invest[0][1]) + '_'
            st.write(text1a)
            text1b='**Number of stocks you should buy:** '+ '_' + str(t_s1) + '_'
            st.write(text1b)
            text1c="**Athena's confidence: **"+'_'+ str(100*invest[0][0])+'%' + '_'
            st.write(text1c)
            st.write('')
            st.write('')

            text2='Your second stock: ' +'_'+ str(invest[1][2])+ '_'
            st.header(text2)
            graphMyStock(invest[1][3],3,4,'springgreen')
            text2a='**Price:** '+ '_$'+ str(invest[1][1])+ '_'
            st.write(text2a)
            text2b='**Number of stocks you should buy:** '+'_'+ str(t_s2)+ '_'
            st.write(text2b)
            text2c="**Athena's confidence:** "+'_'+ str(100*invest[1][0]) + '%'+'_'
            st.write(text2c)
            st.write('')
            st.write('')

            text3= 'Your third stock: '+'_'+ str(invest[2][2])+ '_'
            st.header(text3)
            graphMyStock(invest[2][3],5,6,'indianred')
            text3a='**Price:** '+ '_$'+  str(invest[2][1])+ '_'
            st.write(text3a)
            text3b='**Number of stocks you should buy: **'+'_'+ str(t_s3)+'_'
            st.write(text3b)
            text3c="**Athena's confidence: **"+'_'+ str(100*invest[2][0]) + '%'+'_'
            st.write(text3c)
            st.write('')
            st.write('')

 

        else:
            st.write('Budget too low to diversify')



if a=='Understand':    
    ticker = st.sidebar.selectbox(
            'Choose a Stock',symbols)
    
    stock = yf.Ticker(ticker)
    info=stock.info



    ln=info['longName']
    st.title(info['longName'])
    st.title(ticker)
            
    opt1, opt2 = st.beta_columns(2)
            
    with opt1:
        numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
            
    with opt2:
        windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  
                

    start = dt.datetime.today()-dt.timedelta(numYearMA * 365)
    end = dt.datetime.today()
    livedata = yf.download(ticker,start,end)
    df_ma = calcMovingAverage(livedata, windowSizeMA)
    df_ma = df_ma.reset_index()
                
    fig = go.Figure()
            
    fig.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma['Adj Close'],
                name = '('+ ticker+ ') '+ "Prices Over Last " + str(numYearMA) + " Year(s)",
                mode='lines',
                line=dict(color='royalblue')
                        )
                )
    compstock2=st.selectbox('Choose stock to compare with: ', symbols)
    st.info("If you don't wish to compare, select the same stock again")
    livedata2=yf.download(compstock2,start,end)
    df_ma2= calcMovingAverage(livedata2, windowSizeMA)
    df_ma2= df_ma2.reset_index()
    fig.add_trace(
        go.Scatter(
                x=df_ma2['Date'],
                y=df_ma2['Adj Close'],
                name = '('+ compstock2+ ') '+ "Prices Over Last " + str(numYearMA) + " Year(s)",
                mode='lines',
                line=dict(color='firebrick')
                    ))



                    
    fig.update_layout(showlegend=True,legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            ))
                
    fig.update_layout(legend_title_text='Trend')
    fig.update_yaxes(tickprefix="$")
    
        
    st.plotly_chart(fig, use_container_width=True)  
            

            
    st.subheader('Bollinger Band')
    opta, optb = st.beta_columns(2)
    with opta:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
                
    with optb:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
            
    startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker,startBoll,endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
                    go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"
                        )
                )
            
            
    figBoll.add_trace(
                        go.Scatter(
                                x = df_boll['Date'],
                                y = df_boll['sma'],
                                name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                            )
                    )
            
            
    figBoll.add_trace(
                        go.Scatter(
                                x = df_boll['Date'],
                                y = df_boll['bold'],
                                name = "Lower Band"
                            )
                    )
            
    figBoll.update_layout(showlegend=True,legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0
            ))
            
    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)
    st.sidebar.title("Stock News")
    send="https://www.google.com/search?q=should+you+invest+in+ "+ln.lower()+" stock"
    res=requests.get(send)
    soup=BeautifulSoup(res.content, "html.parser")
    all_links=[]
    all_titles=[]
    count=0
    for i in soup.select("a"):
        if count==5:
            break
        link=i.get("href")
        if("/url?q=https://" in link):
            if(("/url?q=https://support.google.com" not in link) and ("/url?q=https://accounts.google.com" not in link)):
                x=link.split("https://")
                y=x[1].split("&sa")
                new="https://"+y[0]
                all_links.append(new)
                z=i.text
                if("..." in z):
                    type2=z.split("...")
                    name=type2[0]
                else:
                    type1=z.split(" › ")
                    name=type1[0]
                all_titles.append(name)
                count+=1
    for i in range(len(all_titles)):
        make="["+str(all_titles[i])+"]"+" "+"("+str(all_links[i])+")"
        st.sidebar.markdown(make)
        st.sidebar.write("")
        st.sidebar.write("")
    
    list1=[]
    c=0
    for i in all_links:
        if c==10:
            break
        option=requests.get(i)
        soup=BeautifulSoup(option.content, "html.parser")
        pageinfo=soup.select("p")
        for j in pageinfo:
            m=j.text
            n=m.split(' ')
            for i in n:
                list1.append(i)
        c=c+1
    
    tex=' '.join(list1)
    understand_prob=predict_class(tex,model)

    finint=understand_prob['intent']
    finprob=100*float(understand_prob['probability'])
    if finint=='up':
        fininta='Stock prices will go up'
    elif finint=='down':
        fininta='Stock prices will go down'
    fina='**Stock trend prediction: **' + '_'+ str(fininta)+ '_'
    finb="**Athena's confidence: **"+ '_'+ str(finprob)+'%' +'_'
    st.subheader(fina)
    st.subheader(finb)