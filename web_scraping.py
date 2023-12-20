from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime

from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy

def wingd_scraper():
    options = webdriver.ChromeOptions()
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 1200")
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    # driver = webdriver.Chrome()
    driver.get('https://wingd.e-vesseltracker.com')
    driver.find_element(By.ID,'user_login').send_keys('vsingh')
    driver.find_element(By.ID,'user_password').send_keys('h4VMFO423C')
    driver.find_element(By.ID,'submit_btn').click()
    driver.find_element(By.LINK_TEXT,'MU LAN').click() #for selecting the vessel
    driver.find_element(By.LINK_TEXT,'Measurements').click() #navigating to measurements tab in home page
    driver.find_element(By.LINK_TEXT,'AMS').click()
    driver.find_element(By.ID,'begin_date').clear()
    driver.find_element(By.ID,'begin_date').send_keys(str(datetime.now().year)+'-'+str(datetime.now().month)+'-'+str(datetime.now().day-1))
    driver.find_element(By.ID,'end_date').clear()
    driver.find_element(By.ID,'end_date').send_keys(str(datetime.now().year)+'-'+str(datetime.now().month)+'-'+str(datetime.now().day-1))
    driver.find_element(By.ID,'submit_btn').click()

    time.sleep(10)
    driver.find_element(By.CLASS_NAME,'popup_box_show_button').click()
    time.sleep(10)
    d = driver.find_element(By.ID,'column_1')
    drop=Select(d).options
    ams_col_list = []
    for i in drop:
        ams_col_list.append(i.text)
    time.sleep(10)
    driver.find_element(By.ID,'close_button').click()
    exclude_col = ams_col_list.copy()
    exclude_col.remove('None')

    exclude_col = ['Fuel oil temperature supply unit (me_1_signals)','Fuel oil temperature supply unit (me_2_signals)','TC Bearing Oil Pressure Inlet TC #01 (me_1_signals)','TC Bearing Oil Pressure Inlet TC #01 (me_2_signals)']


    driver.find_element(By.XPATH,"//td[@class='popup_box_show_button']").click()
    time.sleep(5)
    for id in range(len(exclude_col)):
        d = driver.find_element(By.ID,'column_'+str(id+1))
        drop=Select(d)
        drop.select_by_index(ams_col_list.index(exclude_col[id]))
    driver.find_element(By.XPATH,"//button[@id='save']").click()
    print('Saved!')
    time.sleep(10) 

    page_content = driver.page_source
    soup = BeautifulSoup(page_content, 'html.parser')
    tbody = soup.find_all('tr', id='tab_content')

    new_list = []
    row_data = pd.DataFrame()
    j = 0
    while j<len(tbody):
        values = []
        values.append(tbody[j].find('td','td_list_row').text)
        cv = tbody[j].find_all('td','td_list_row_center')
        for column_idx in range(len(cv)):
            values.append(cv[column_idx].text)
        new_list.append(values)
        j+=1
    row_data = pd.concat([row_data,pd.DataFrame(new_list)])
    row_data = row_data.loc[:,:4]
    row_data.columns = ['signaldate']+exclude_col

    row_data.set_index(row_data['signaldate'],drop=True,inplace=True)
    row_data.index = pd.to_datetime(row_data.index)
    row_data = row_data.resample('1H').bfill()
    row_data = row_data.bfill()
    row_data.drop(columns=['signaldate'],inplace=True)
    return row_data
# # driver = web_driver().Chrome
# !pip install keyring
# import keyring as kr 
# kr.set_password("gd","vsingh","h4VMFO423C@123")

# print(cd = kr.get_password("gd","vsingh"))