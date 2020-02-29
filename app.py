# predictor_app.py
import os
import flask
from flask import request,flash, redirect, url_for
from werkzeug.utils import secure_filename
from eda_api import exploratorydataanalysis #, make_prediction 
from eda_api import exploratorydataanalysis
#from gevent.pywsgi import WSGIServer

#UPLOAD_FOLDER = '/uploads'


# Initialize the app
app = flask.Flask(__name__)

app.config.update(

    #Set the secret key to a sufficiently random value
    SECRET_KEY=os.urandom(24),
    #Set the session cookie to be secure
    SESSION_COOKIE_SECURE=True,
    #Set the session cookie for our app to a unique name
    SESSION_COOKIE_NAME='YourAppName-WebSession',
    #Set CSRF tokens to be valid for the duration of the session. This assumes you’re using WTF-CSRF protection
    WTF_CSRF_TIME_LIMIT=None,
    UPLOAD_FOLDER = 'uploads',
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
)



# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/)
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route("/", methods=["GET","POST"])
def upload_file():
    print("in upload file")
    """if request.method == 'POST':
        print("in POST")
        # check if the post request has the file part
        print("request file:",request.files)
        if 'file' not in request.files:
            print("#1")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        file = 'breast-cancer-data.csv'
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("#2")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("#3")
            print("in allowed file")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x_input = exploratorydataanalysis(filename)
           
            #return redirect(url_for('upload_file', filename=filename))

        return flask.render_template('predictor_2.html',chat_in=x_input)"""
    if request.method == 'POST':
       print("in POST")
       filename = 'breast-cancer-data.csv'
       x_input = exploratorydataanalysis(filename)
       return flask.render_template('predictor_2.html',chat_in=x_input)
    if request.method == 'GET':
       print("#4")
       return flask.render_template('predictor_1.html',chat_in=' ')

#def predict():
#    print(request.args)
#    if(request.args):
#        x_input = make_prediction(request.args['chat_in'])
#        print(x_input)
#        return flask.render_template('predictor_2.html',chat_in=x_input)
#    else: 
#        return flask.render_template('predictor_2.html',chat_in=' ')
 


    # Start the server, continuously listen to requests.
if __name__=="__main__":
    # For local development, set to True:
    app.secret_key = os.urandom(50)
    app.run(use_reloader=True,debug=False)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
