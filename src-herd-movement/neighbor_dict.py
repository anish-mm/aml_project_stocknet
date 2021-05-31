"""
This file contains the hard-coded dictionary mapping
each company to its nearest neighbor. For now, we look
at only the 1-NN with Dynamic Time Warping distance 
between the closing prices. Only the reduced 16-company
dataset will be used.
"""

company_list = [
    'AAPL',
    'PG',
    'BUD',
    'KO',
    'AMZN',
    'WMT',
    'CMCSA',
    'HD',
    'BCH',
    'BSAC',
    'BRK-A',
    'JPM',
    'GOOG',
    'MSFT',
    'FB',
    'T',
]

nbr_dict = {"AAPL": "AMZN", "PG": "KO", "BUD": "KO", "KO": "PG", "AMZN": "MSFT", "WMT": "BCH", "CMCSA": "JPM", "HD": "FB", "BCH": "BSAC", "BSAC": "BCH", "BRK-A": "CMCSA", "JPM": "CMCSA", "GOOG": "MSFT", "MSFT": "FB", "FB": "CMCSA", "T": "HD"}