# Imports
import sys
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

# Chrome and Chromedriver paths
# chrome_path = "./bin/chrome/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
# driver_path = "./bin/chromedriver/chromedriver-mac-arm64/chromedriver"

# For Windows, you can use the following paths:
chrome_path = r"C:\tools\chrome-win64\chrome-win64\chrome.exe"
driver_path = r"C:\tools\chromedriver-win64\chromedriver-win64\chromedriver.exe"

def check_or_fail():

    if not os.path.exists(chrome_path):
        print(f"❌ Chrome not found at: {chrome_path}")
        sys.exit(1)

    if not os.path.exists(driver_path):
        print(f"❌ Chromedriver not found at: {driver_path}")
        sys.exit(1)

    print("✅ Chrome and Chromedriver found. Environment ready.")


# Check if chrome and chromedriver are present
check_or_fail()

# Since we are using the chrome and chromedriver for testing, 
# we have to set custom paths for selenium to find these.
options = Options()
options.binary_location = chrome_path

service = Service(driver_path)


# Setting up the environment

# We need to disable automation flags so that selenium can load chrome Dino.

driver = webdriver.Chrome(service=service, options=options)


# Since Selenium returns a 
try:
    driver.get("chrome://dino")
except WebDriverException as e:
    if "net::ERR_INTERNET_DISCONNECTED" in str(e):
        print("⚠️  Expected error: Dino game loaded offline.")
    else:
        raise 
input("Press Enter to close the browser...")

driver.quit()