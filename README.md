# ST-LLM Chatbot
Have a chat with the **deployed** version of the agent [here](https://stllmchat.streamlit.app/)

# Installation Steps
## Windows
1) Clone the contents of the repository.
2) Create a virtual environment using venv <venv name>
3) Activate it using `source venv/Scripts/activate` if you're using Git Bash, `<venv name>/Scripts/activate` otherwise.
4) Download all necessary packages using `pip install -r requirements.txt`
5) To run the agent application, run the command `streamlit run app.py` in the terminal, this should open a new tab in your browser where the app is hosted.

# Evaluation Pipeline
1) To run the evaluation pipeline, with bash run: `bash evaluate.sh`. Expect google tabs to open in the late stage of the process as Selenium is used to evaluate ChatGPT's responses.
2) Run `python analysis.py`
3) With this, you should have all the evaluation data required to reproduce the report.