import csv
import os
from traceback import print_exc
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog

def getfilepath():
    app = QApplication.instance() or QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, "Open video")
    return file_path

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_input(t, prompt):
    while True:
        inputs = input(prompt)
        try:
            return t(inputs)
        except Exception as error:
            print(error)
            print('...Try again')

def csvtodict(path):
    with open(path, 'r') as f:
        dictionary = {row[0]: row[1] for row in csv.reader(f)}
        for key in dictionary.keys():
            try:
                dictionary[key] = float(dictionary[key])
                if dictionary[key] == round(dictionary[key]):
                    dictionary[key] = int(dictionary[key])
            except:
                pass
    return dictionary

def load_settings(program, default_settings):
    print('Loading settings...')
    try:
        settings = csvtodict('settings_' + program + '.csv')
        if settings.keys() != default_settings.keys():
            t = 1/0
        return settings
    except Exception:
        print_exc()
        print('Cannot load settings. Default settings are used.')
        with open('settings_' + program + '.csv', 'w') as f:
            for key in default_settings.keys():
                f.write(key + ',' + str(default_settings[key]) + '\n')
        print('settings_' + program + '.csv has been produced using default values.')
        if input('Enter e to exit the program to change settings, others to continue with default settings:') == 'e':
            sys.exit()
        return default_settings
