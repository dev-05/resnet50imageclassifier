from flask import Flask, render_template, request, redirect, flash , url_for
import mypredict
import urllib.request
from werkzeug.utils import secure_filename
from mypredict import getpredictions
import os


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = '/home/dev/Desktop/googleclone/Uploaded_temp'

@app.route('/')
def index():
    return render_template('myindex.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('Uploaded_temp',filename))
            getpredictions(filename)
            k_notuseful,label, acc = getpredictions(filename)
            label=str(label)
            acc=str(round(acc*100,2))
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()
