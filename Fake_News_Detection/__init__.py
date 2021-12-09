"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Fake_News_Detection.views
