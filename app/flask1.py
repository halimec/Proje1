#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,redirect,url_for,render_template,request
from flask import request
import pickle
from flask import render_template, render_template_string, request


# In[2]:


app=Flask(__name__)


# In[3]:


@app.route("/")
def main():
    return render_template("index.html")
    
   


# In[4]:


@app.route("/page2")
def page2():
    return render_template("page2.html")
    


# In[5]:


@app.route("/page3",methods=["POST","GET"])
def page3():
    il = request.form["il"]
    ilce = request.form["ilce"]
    #mah = request.form["mah"]
    oda = request.form["OdaSayısı"]
    binayasi = request.form["BinanınYASI"]
    bulundugukat = request.form["BulunduguKAT"]
    binadakikat = request.form["BinadakiKatSAYISI"]
    ısıtma = request.form["IsıtmaTipi"]
    banyo = request.form["BanyoSAYISI"]
    
    
    with open('nisan2gaussianpkl.pkl' , 'rb') as f:
        lr = pickle.load(f)
    with open('nisan2svmmodelpkl.pkl' , 'rb') as f:
        lr1 = pickle.load(f)
    with open('nisan2linearpkl.pkl' , 'rb') as f:
        lr2 = pickle.load(f)
    
    il=int(il)
    ilce=int(ilce)
    #mah=int(mah)
    oda=int(oda)
    binayasi=int(binayasi)
    bulundugukat=int(bulundugukat)
    binadakikat=int(binadakikat)
    ısıtma=int(ısıtma)
    banyo=int(banyo)
    
    
    List = [il, ilce, oda,binayasi,bulundugukat,binadakikat,ısıtma,banyo]
    
    
    tahmin=lr.predict([List])
    tahmin1=lr1.predict([List])
    tahmin2=lr2.predict([List])
    
    
    return redirect(url_for("user",tahmin=tahmin,tahmin1=tahmin1,tahmin2=tahmin2))
    
    
    
    


# In[6]:


@app.route("/<tahmin>/<tahmin1>/<tahmin2>")
def user(tahmin,tahmin1,tahmin2):
    if request.method == "GET":
        return render_template ("page2.html",tahmin=tahmin,tahmin1=tahmin1,tahmin2=tahmin2)
    if request.method == "POST":
        return redirect(request.referrer)
    


# In[7]:


if __name__== "__main__":
    app.run()


# In[ ]:




