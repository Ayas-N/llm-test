from flask import Blueprint, render_template, request, redirect, url_for, session, make_response, send_from_directory, send_file, flash
import config
import os
from mem0 import Memory

os.environ["OPENAI_API_KEY"] = open(file= 'apikey.txt').read()


MAX_TOKENS = 1024

app_blueprint =  Blueprint('app_blueprint', __name__)

@app_blueprint.route("/")
def home():
    return render_template("main.html")

@app_blueprint.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    response = openai.chat.completions.create(
        messages = [{
            "role":"user",
            "content":"Say this is a test",
        }],
        model= "gpt-4o-mini",
    )

    answer = response.choices[0].message.content
    return str(answer)
