from flask import Flask,render_template,url_for,request
from main import input_process,recommend_movie,df

app = Flask(__name__)


def send_info(ids):
    sd={}
    j=[] 
    for i in ids:
        m_name = df['title'][i]
        year = df['year'][i]
        rating = df['rating'][i]
        summary = df['summary'][i]
        imdb = f"https://www.imdb.com/title/{str(df['imdb_id'][i])}/?ref_=fn_al_tt_1"
        sd[m_name] = [year,rating,summary,imdb]
    names=[]
    for i in ids:
        names.append(df['title'][i])
    return sd,names





@app.route('/',methods=['GET','POST'])
def Home():
    if request.method == 'POST':
        check = False
        desc = request.form['input_field']
        processed_inp = input_process(desc)
        ids = recommend_movie(processed_inp)
        if len(ids) > 1:
            check =True
            dic,Movie_names = send_info(ids)
            return render_template('index.html',dic=dic,Movie_names=Movie_names,check=check)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()