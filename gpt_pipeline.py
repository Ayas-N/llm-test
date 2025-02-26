import time
import prompts
from selenium import webdriver
from seleniumbase import Driver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os

driver = Driver(uc = True)
time.sleep(3)


driver.get("https://chatgpt.com")
wait = WebDriverWait(driver , 10)
driver.get("https://chatgpt.com")
time.sleep(2)

def generate_with_agent(algorithm):
    usr_input = f"""Please summarise the {algorithm.upper()} algorithm from spatial transcriptomics with the following properties, 
    I have given you a list of properties I wish for you to include in your summary below:\n""" + prompts.response_struct
    input_box = driver.find_element(by=By.ID, value = "prompt-textarea")

    for line in usr_input.split("\n"):
        input_box.send_keys(line)
        input_box.send_keys(Keys.SHIFT, Keys.RETURN)

    input_box.submit()
    time.sleep(60)
    messages = driver.find_elements(by=By.CSS_SELECTOR, value='div.markdown.prose.w-full.break-words.dark\\:prose-invert.light')
    latest_message = messages[-1].text
    print(latest_message)
    # while True:
    #     # Y continues evaluation on more papers, N refreshes the latest message.
    #     # You will need to click "Stay logged out" after the first 5 uses.
    #     cont = input("Would you like to continue? (Y/N)\n")
    #     if cont.lower() == "y": break 
    #     if cont.lower() == "n": 
    #         messages = driver.find_elements(by=By.CSS_SELECTOR, value='div.markdown.prose.w-full.break-words.dark\\:prose-invert.light')
    #         latest_message = messages[-1].text

    return latest_message

pdfs = [os.path.splitext(filename)[0] for filename in os.listdir("pdfs")]
for i in range(1,6):
    for algo in pdfs:
        with open(f"sim{i}/gpt_out/{algo}.csv", "w") as f:
            f.write(generate_with_agent(algo))