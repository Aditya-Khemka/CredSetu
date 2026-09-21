"""Single-user UPI transaction generator (v2: ledger-based, with balances) -- the single source of truth for generation logic.

Derived from upi-transactions-generator.ipynb: hourly_distribution() and get_amt() are copied verbatim from the
original Kaggle generator; everything after them is the single-user extension.

v2 adds a hidden per-customer bank ledger: every emitted row is a UPI debit or credit; salary/ATM/one-off bank flows
live only in the ledger and show up as balance changes between rows. Latent parameters are never emitted.

Use:  from upi_user_generator import generate_user_transactions, verify_user_history
      python driver.py   (or)   python upi_user_generator.py [n_records]   ->  writes user001.csv
"""
import heapq
import itertools
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------- Original generator functions (unchanged) ----------------
def hourly_distribution():
    '''Return realistic hourly distribution for UPI transactions'''
    # Peak hours: 10-13 and 17-21
    hours_prob = [0.013942,0.009293,0.007435,0.004647,0.004647,0.007435,0.013941,0.023234,0.032528,0.041822,0.055762,0.065056,0.069703,0.060409,  
                  0.046468,0.051115,0.055762,0.074349,0.078996,0.083643,0.074349,0.065056,0.037175,0.023234,]
    rounded = np.round(hours_prob, 3)
    rounded[np.argmax(rounded)] += 1.0 - rounded.sum()
    return rounded


def get_amt(merchant_categories, age_groups):
    '''Generate realistic transaction amounts based on merchant category and age group'''
    amounts = []
    
    # Age-based spending multipliers for each category
    age_multipliers = {
        '18-25': {
            'Food': 1.15,           # Young people dine out more frequently
            'Grocery': 0.85,        # Smaller household needs
            'Fuel': 0.90,          # Less vehicle ownership
            'Entertainment': 1.30,  # High entertainment spending
            'Shopping': 1.25,       # Fashion and lifestyle focus
            'Healthcare': 0.60,     # Minimal health issues
            'Education': 0.70,      # Personal learning, not family education
            'Transport': 1.20,      # High mobility, ride-sharing
            'Utilities': 0.70,      # Shared living, lower bills
            'Other': 1.10          # Miscellaneous young adult spending
        },
        '26-35': {
            'Food': 1.10,           # Continued dining out but moderating
            'Grocery': 1.05,        # Establishing households
            'Fuel': 1.00,          # Standard vehicle usage
            'Entertainment': 1.15,  # Still active entertainment spending
            'Shopping': 1.15,       # Fashion and lifestyle purchases
            'Healthcare': 0.85,     # Increasing but still moderate
            'Education': 1.20,      # Professional development, early parenting
            'Transport': 1.10,      # Commuting and lifestyle
            'Utilities': 0.95,      # Moderate household expenses
            'Other': 1.05          # General adult spending
        },
        '36-45': {
            'Food': 1.00,           # Balanced family dining
            'Grocery': 1.20,        # Peak family shopping needs
            'Fuel': 1.15,          # Family vehicle usage
            'Entertainment': 0.95,  # Family-oriented entertainment
            'Shopping': 1.00,       # Household and family needs
            'Healthcare': 1.15,     # Increasing health consciousness
            'Education': 1.50,      # Peak children's education spending
            'Transport': 1.05,      # Family transport needs
            'Utilities': 1.20,      # Full household utility expenses
            'Other': 1.00          # Standard family spending
        },
        '46-55': {
            'Food': 0.95,           # Reduced dining out
            'Grocery': 1.15,        # Continued family needs
            'Fuel': 1.10,          # Maintained vehicle usage
            'Entertainment': 0.80,  # Reduced entertainment spending
            'Shopping': 0.85,       # Reduced fashion spending
            'Healthcare': 1.40,     # Significant increase in health spending
            'Education': 1.30,      # Continued children's education
            'Transport': 0.95,      # Reduced leisure travel
            'Utilities': 1.35,      # Higher utility consumption
            'Other': 0.90          # More conservative spending
        },
        '56+': {
            'Food': 0.85,           # Limited dining out
            'Grocery': 1.00,        # Stable grocery needs
            'Fuel': 0.80,          # Reduced mobility
            'Entertainment': 0.65,  # Minimal entertainment spending
            'Shopping': 0.70,       # Essential purchases only
            'Healthcare': 2.20,     # Highest healthcare spending
            'Education': 0.80,      # Grandchildren or personal interest
            'Transport': 0.75,      # Limited transport needs
            'Utilities': 1.50,      # Higher utility usage due to home presence
            'Other': 0.80          # Conservative miscellaneous spending
        }
    }
    
    for i, category in enumerate(merchant_categories):
        age_group = age_groups[i] if isinstance(age_groups, (list, np.ndarray)) else age_groups
        
        if category == 'Food':
            # Food/Restaurant: 50-2000, concentrated around 200-800
            rand = np.random.random()
            if rand < 0.40:
                amount = np.random.lognormal(mean=np.log(120), sigma=0.6)  # 50-200
            elif rand < 0.80:
                amount = np.random.lognormal(mean=np.log(400), sigma=0.5)  # 200-800
            else:
                amount = np.random.lognormal(mean=np.log(1200), sigma=0.4)  # 800-2000
                
        elif category == 'Grocery':
            # Grocery: 100-3000, concentrated around 400-1500
            rand = np.random.random()
            if rand < 0.30:
                amount = np.random.lognormal(mean=np.log(250), sigma=0.5)  # 100-400
            elif rand < 0.70:
                amount = np.random.lognormal(mean=np.log(800), sigma=0.6)  # 400-1500
            else:
                amount = np.random.lognormal(mean=np.log(2000), sigma=0.4)  # 1500-3000
                
        elif category == 'Fuel':
            # Fuel: 300-5000, based on vehicle type
            rand = np.random.random()
            if rand < 0.40:
                amount = np.random.lognormal(mean=np.log(500), sigma=0.3)  # 300-800 (two-wheeler)
            elif rand < 0.70:
                amount = np.random.lognormal(mean=np.log(1200), sigma=0.4)  # 800-2000 (car partial)
            else:
                amount = np.random.lognormal(mean=np.log(3000), sigma=0.3)  # 2000-5000 (full tank)
                
        elif category == 'Entertainment':
            # Entertainment: 100-1500, concentrated around movie/event tickets
            rand = np.random.random()
            if rand < 0.50:
                amount = np.random.lognormal(mean=np.log(200), sigma=0.4)  # 100-400 (tickets)
            elif rand < 0.80:
                amount = np.random.lognormal(mean=np.log(600), sigma=0.5)  # 400-1000 (events)
            else:
                amount = np.random.lognormal(mean=np.log(300), sigma=0.6)  # 100-500 (gaming)
                
        elif category == 'Shopping':
            # Shopping: 200-8000, wide range for different items
            rand = np.random.random()
            if rand < 0.40:
                amount = np.random.lognormal(mean=np.log(500), sigma=0.7)  # 200-1000
            elif rand < 0.75:
                amount = np.random.lognormal(mean=np.log(1800), sigma=0.6)  # 1000-3000
            else:
                amount = np.random.lognormal(mean=np.log(5000), sigma=0.4)  # 3000-8000
                
        elif category == 'Healthcare':
            # Healthcare: 200-2000, consultation and medicine focused
            rand = np.random.random()
            if rand < 0.50:
                amount = np.random.lognormal(mean=np.log(400), sigma=0.5)  # 200-800 (consultations)
            elif rand < 0.80:
                amount = np.random.lognormal(mean=np.log(300), sigma=0.4)  # 100-500 (medicines)
            else:
                amount = np.random.lognormal(mean=np.log(1000), sigma=0.5)  # 500-2000 (tests)
                
        elif category == 'Education':
            # Education: 500-15000, varied based on course type
            rand = np.random.random()
            if rand < 0.40:
                amount = np.random.lognormal(mean=np.log(1500), sigma=0.6)  # 500-3000 (online)
            elif rand < 0.75:
                amount = np.random.lognormal(mean=np.log(4000), sigma=0.5)  # 2000-8000 (tuition)
            else:
                amount = np.random.lognormal(mean=np.log(8000), sigma=0.4)  # 5000-15000 (institutional)
                
        elif category == 'Transport':
            # Transport: 30-800, based on ride type and distance
            rand = np.random.random()
            if rand < 0.50:
                amount = np.random.lognormal(mean=np.log(80), sigma=0.6)  # 30-200 (short rides)
            else:
                amount = np.random.lognormal(mean=np.log(400), sigma=0.6)  # 100-800 (app cabs)
                
        elif category == 'Utilities':
            # Utilities: 200-8000, concentrated around typical bill amounts
            rand = np.random.random()
            if rand < 0.25:
                amount = np.random.lognormal(mean=np.log(300), sigma=0.4)  # 100-500 (water)
            else:
                amount = np.random.lognormal(mean=np.log(2500), sigma=0.6)  # 800-5000 (electricity)
                
        else:  # Other category
            # Other: 100-2000, general miscellaneous payments
            amount = np.random.lognormal(mean=np.log(600), sigma=0.8)  # 100-2000

        # Apply age-based multiplier
        multiplier = age_multipliers.get(age_group, {}).get(category, 1.0)
        amount *= multiplier
        
        # Ensure minimum amount and round to nearest rupee
        amount = max(amount, 10)  # Minimum 10
        amount = min(amount, 100000)  # UPI limit 1 lakh
        amounts.append(round(amount))
    
    return np.array(amounts)


# ---------------- Single-user extension ----------------
# ---------------- Persistent recurring-payment configuration ----------------
# service: (Merchant_Category, transaction type, typical plan prices in INR or None if the bill varies, billing period)
RECURRING_SERVICES = {
    # Entertainment (subscriptions, monthly)
    'Netflix':         ('Entertainment', 'P2M',          [199, 499, 649], 'monthly'),
    'Spotify':         ('Entertainment', 'P2M',          [59, 119],       'monthly'),
    'Prime Video':     ('Entertainment', 'P2M',          [299],           'monthly'),
    'YouTube Premium': ('Entertainment', 'P2M',          [149],           'monthly'),
    'JioHotstar':      ('Entertainment', 'P2M',          [149, 299],      'monthly'),
    # Telecom (prepaid plans, 28-day validity). A user has at most one operator.
    'Airtel':          ('Telecom',       'Recharge',     [299, 349, 479], '28d'),
    'Jio':             ('Telecom',       'Recharge',     [239, 299, 349], '28d'),
    'Vi':              ('Telecom',       'Recharge',     [299, 349],      '28d'),
    'BSNL':            ('Telecom',       'Recharge',     [199, 229],      '28d'),
    # Utilities (monthly bills). The receiver name is the user's actual provider (see PROVIDERS below).
    'Electricity':     ('Utilities',     'Bill Payment', None,            'monthly'),  # varies each month
    'Internet':        ('Utilities',     'Bill Payment', [499, 799, 999], 'monthly'),
    'Gas':             ('Utilities',     'Bill Payment', None,            'monthly'),  # varies each month
}
TELECOM_OPERATORS = ['Airtel', 'Jio', 'Vi', 'BSNL']

# Probability that a user has each recurring relationship, by age group.
# Telecom operators are mutually exclusive (sum < 1 => some users have no recurring recharge).
recurring_probability_by_age = {
    '18-25': {'Netflix': 0.30, 'Spotify': 0.35, 'Prime Video': 0.25, 'YouTube Premium': 0.25, 'JioHotstar': 0.30,
              'Airtel': 0.25, 'Jio': 0.45, 'Vi': 0.10, 'BSNL': 0.02,
              'Electricity': 0.10, 'Internet': 0.15, 'Gas': 0.05},
    '26-35': {'Netflix': 0.40, 'Spotify': 0.30, 'Prime Video': 0.35, 'YouTube Premium': 0.20, 'JioHotstar': 0.30,
              'Airtel': 0.30, 'Jio': 0.40, 'Vi': 0.12, 'BSNL': 0.03,
              'Electricity': 0.45, 'Internet': 0.40, 'Gas': 0.25},
    '36-45': {'Netflix': 0.30, 'Spotify': 0.15, 'Prime Video': 0.30, 'YouTube Premium': 0.10, 'JioHotstar': 0.30,
              'Airtel': 0.30, 'Jio': 0.38, 'Vi': 0.12, 'BSNL': 0.05,
              'Electricity': 0.65, 'Internet': 0.40, 'Gas': 0.45},
    '46-55': {'Netflix': 0.15, 'Spotify': 0.08, 'Prime Video': 0.20, 'YouTube Premium': 0.05, 'JioHotstar': 0.25,
              'Airtel': 0.28, 'Jio': 0.33, 'Vi': 0.12, 'BSNL': 0.10,
              'Electricity': 0.75, 'Internet': 0.30, 'Gas': 0.55},
    '56+':   {'Netflix': 0.08, 'Spotify': 0.03, 'Prime Video': 0.12, 'YouTube Premium': 0.03, 'JioHotstar': 0.15,
              'Airtel': 0.25, 'Jio': 0.28, 'Vi': 0.12, 'BSNL': 0.15,
              'Electricity': 0.80, 'Internet': 0.20, 'Gas': 0.55},
}

# Real provider behind the generic recurring utility services (depends on the user's home state)
ELECTRICITY_PROVIDERS = {
    'Maharashtra': ['MSEDCL', 'Adani Electricity', 'Tata Power'], 'Karnataka': ['BESCOM'],
    'Delhi': ['BSES Rajdhani', 'BSES Yamuna', 'Tata Power-DDL'], 'Tamil Nadu': ['TANGEDCO'],
    'West Bengal': ['CESC', 'WBSEDCL'], 'Gujarat': ['Torrent Power', 'UGVCL'],
    'Rajasthan': ['JVVNL', 'AVVNL'], 'Uttar Pradesh': ['UPPCL', 'Noida Power'],
    'Andhra Pradesh': ['APSPDCL', 'APEPDCL'], 'Telangana': ['TSSPDCL', 'TSNPDCL']}
GAS_PROVIDERS = {'Maharashtra': ['Mahanagar Gas'], 'Delhi': ['Indraprastha Gas'], 'Gujarat': ['Gujarat Gas', 'Adani Total Gas']}
DEFAULT_GAS_PROVIDERS = ['Indane Gas', 'HP Gas', 'Bharat Gas']  # LPG cylinder booking where there is no piped gas
INTERNET_PROVIDERS = ['Airtel Xstream Fiber', 'JioFiber', 'ACT Fibernet', 'Hathway', 'BSNL Bharat Fiber']

# Small age-based nudges to the P2M merchant-category weights (1.0 = original weight)
age_category_weight_adjustment = {
    '18-25': {'Food': 1.10, 'Entertainment': 1.15, 'Shopping': 1.10, 'Transport': 1.10, 'Healthcare': 0.80},
    '26-35': {'Food': 1.05, 'Entertainment': 1.05, 'Education': 1.10, 'Healthcare': 0.90},
    '36-45': {'Grocery': 1.10, 'Education': 1.20, 'Fuel': 1.05},
    '46-55': {'Healthcare': 1.30, 'Entertainment': 0.90, 'Shopping': 0.90},
    '56+':   {'Healthcare': 1.60, 'Entertainment': 0.75, 'Shopping': 0.80, 'Transport': 0.85},
}

# Original merchant categories / weights (Utilities is now reached via Bill Payment, Telecom via Recharge)
P2M_CATEGORIES = ['Food', 'Grocery', 'Fuel', 'Entertainment', 'Shopping', 'Healthcare', 'Education', 'Transport', 'Other']
P2M_BASE_WEIGHTS = np.array([0.15, 0.20, 0.10, 0.08, 0.12, 0.05, 0.03, 0.08, 0.10])

BANKS = ['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'Kotak', 'IndusInd', 'Yes Bank']
BANK_P = [0.25, 0.15, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10]
STATES = ['Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'West Bengal',
          'Gujarat', 'Rajasthan', 'Uttar Pradesh', 'Andhra Pradesh', 'Telangana']
STATE_P = [0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.12, 0.08, 0.09]

# ---------------- Receivers ----------------
# category: (well-known merchants ordered roughly by popularity, templates for local shops ({} = a surname),
#            (min, max) brands the user actually uses, (min, max) local shops the user uses)
# Subscription / recurring services (Netflix, Airtel, ...) are deliberately NOT in these pools: they only appear
# through the recurring mechanism, so repeated payments to them are genuine recurring relationships.
MERCHANT_CATALOG = {
    'Food': (['Zomato', 'Swiggy', "Domino's Pizza", "McDonald's", 'KFC', 'Starbucks', 'Burger King', 'Chai Point',
              'Subway', "Haldiram's", 'Pizza Hut', 'Barbeque Nation', 'Behrouz Biryani', 'Faasos', 'Box8', 'Wow! Momo',
              'Cafe Coffee Day', 'Theobroma', 'Baskin Robbins', 'Third Wave Coffee', 'EatSure', "Dunkin'"],
             ['{} Restaurant', 'Hotel {}', '{} Sweets', '{} Bakery', '{} Tiffin Centre', '{} Fast Food',
              'Juice Corner', 'Chai Tapri', '{} Biryani House'], (6, 12), (3, 6)),
    'Grocery': (['Blinkit', 'Zepto', 'BigBasket', 'Swiggy Instamart', 'DMart', 'JioMart', 'Reliance Fresh',
                 'More Supermarket', 'Amazon Fresh', 'Country Delight', 'Licious', 'Spencers', "Nature's Basket",
                 'Star Bazaar', 'Milkbasket', 'FreshToHome', 'Vishal Mega Mart', 'Easyday', 'Ratnadeep Supermarket',
                 'Nilgiris', 'Spar Hypermarket', 'Modern Bazaar', 'Heritage Fresh', 'Lulu Hypermarket'],
                ['{} Kirana Store', '{} General Store', '{} Provision Store', '{} Supermarket', '{} Dairy',
                 '{} Fruits & Vegetables', '{} Traders', '{} Departmental Store', 'Sri {} Stores'], (8, 16), (3, 6)),
    'Fuel': (['Indian Oil', 'HP Petrol Pump', 'Bharat Petroleum', 'Shell', 'Nayara Energy', 'Jio-bp'],
             ['{} Filling Station', '{} Petroleum'], (2, 3), (0, 1)),
    'Entertainment': (['BookMyShow', 'PVR INOX', 'Cinepolis', 'Dream11', 'Google Play', 'Paytm Insider', 'MPL',
                       'Steam', 'PlayStation Store', 'Wonderla', 'District by Zomato', 'Imagica'],
                      ['{} Gaming Zone', '{} Multiplex', 'Fun City Arcade', '{} Bowling Alley'], (3, 6), (0, 2)),
    'Shopping': (['Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Nykaa', 'Meesho', 'Croma', 'Decathlon', 'Reliance Digital',
                  'Pantaloons', 'Lifestyle', 'Westside', 'Max Fashion', 'Shoppers Stop', 'Lenskart', 'Tata CLiQ',
                  'FirstCry', 'IKEA', 'Zara', 'H&M', 'Uniqlo', 'Vijay Sales', 'Bata', 'Titan', 'Pepperfry'],
                 ['{} Garments', '{} Electronics', '{} Footwear', '{} Mobile Store', '{} Fashion Hub',
                  '{} Jewellers', '{} Gift Shop'], (8, 14), (2, 4)),
    'Healthcare': (['Apollo Pharmacy', 'MedPlus', 'PharmEasy', 'Tata 1mg', 'Netmeds', 'Practo', 'Dr Lal PathLabs',
                    'Thyrocare', 'Metropolis Labs', 'Apollo Hospitals', 'Fortis Healthcare', 'Max Healthcare',
                    'Manipal Hospitals'],
                   ['{} Medical Store', 'Dr. {} Clinic', '{} Diagnostics', '{} Dental Care', '{} Nursing Home',
                    '{} Eye Care'], (3, 6), (2, 3)),
    'Education': (['Udemy', 'Coursera', 'Unacademy', 'upGrad', "BYJU'S", 'Physics Wallah', 'Simplilearn', 'Vedantu',
                   'Great Learning', 'LinkedIn Learning', 'edX', 'British Council'],
                  ['{} Coaching Classes', '{} Tutorials', '{} Institute', '{} Academy'], (2, 4), (0, 2)),
    'Transport': (['Uber', 'Ola', 'Rapido', 'IRCTC', 'redBus', 'NHAI FASTag', 'Metro Card Recharge', 'BluSmart',
                   'inDrive', 'Yulu', 'Indian Railways UTS', 'abhiBus'],
                  ['{} Travels', '{} Auto Services'], (3, 6), (0, 1)),
    'Other': (['Urban Company', 'Cult.fit', 'India Post', 'DTDC Courier', 'Dunzo', 'Porter', 'Blue Dart'],
              ['{} Salon', '{} Laundry & Dry Cleaners', '{} Stationers', '{} Gym', '{} Print & Xerox',
               '{} Hardware Store', '{} Florists', '{} Pet Care', 'Sri {} Enterprises'], (2, 4), (3, 5)),
    # Organic (non-recurring) Bill Payment receivers
    'Utilities': (['Tata Play', 'Dish TV', 'Airtel Digital TV', 'Sun Direct', 'Municipal Water Board',
                   'Municipal Corporation Property Tax', 'LIC of India', 'HDFC Life', 'ICICI Prudential Life',
                   'Star Health Insurance', 'Bajaj Allianz Insurance', 'Apartment Society Maintenance'],
                  [], (2, 4), (0, 0)),
}

FIRST_NAMES = ['Aarav', 'Rohan', 'Vikram', 'Arjun', 'Karthik', 'Rahul', 'Amit', 'Sanjay', 'Manish', 'Deepak', 'Suresh',
               'Nikhil', 'Varun', 'Aditya', 'Harsh', 'Sagar', 'Ravi', 'Prakash', 'Ankit', 'Siddharth',
               'Priya', 'Ananya', 'Neha', 'Pooja', 'Sneha', 'Kavya', 'Divya', 'Meera', 'Ritu', 'Anjali',
               'Sunita', 'Lakshmi', 'Shreya', 'Isha', 'Nisha', 'Swati', 'Rekha', 'Aishwarya', 'Tanvi', 'Komal']
LAST_NAMES = ['Sharma', 'Verma', 'Patel', 'Reddy', 'Nair', 'Iyer', 'Singh', 'Gupta', 'Kumar', 'Joshi', 'Mehta',
              'Shah', 'Desai', 'Rao', 'Pillai', 'Das', 'Banerjee', 'Chatterjee', 'Mishra', 'Yadav', 'Agarwal',
              'Jain', 'Kulkarni', 'Naidu', 'Menon', 'Bose', 'Chauhan', 'Thakur', 'Pandey', 'Kapoor', 'Malhotra',
              'Gowda', 'Hegde', 'Bhat', 'Saxena', 'Tiwari', 'Choudhary', 'Patil', 'Deshmukh', 'Rathore']


def random_person_name():
    return f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"


def zipf_weights(n, s_low=0.8, s_high=1.4):
    '''Skewed usage weights: a few favourites, then a long tail. The skew itself is drawn per user.'''
    w = 1.0 / np.arange(1, n + 1) ** np.random.uniform(s_low, s_high)
    return w / w.sum()


def build_receiver_pool(category, exclude=()):
    '''The merchants this user actually pays in one category: some well-known brands (popular ones more likely)
    plus a few local shops, with persistent favourites. Fuel has few options, Grocery/Food/Shopping many.'''
    brands, templates, (b_lo, b_hi), (l_lo, l_hi) = MERCHANT_CATALOG[category]
    brands = [b for b in brands if b not in exclude]
    k = min(int(np.random.randint(b_lo, b_hi + 1)), len(brands))
    popularity = 1.0 / (np.arange(len(brands)) + 4)
    pool = [str(b) for b in np.random.choice(brands, size=k, replace=False, p=popularity / popularity.sum())]
    if templates:
        for _ in range(int(np.random.randint(l_lo, l_hi + 1))):
            name = str(np.random.choice(templates)).format(np.random.choice(LAST_NAMES))
            if name not in pool:
                pool.insert(int(np.random.randint(0, len(pool) + 1)), name)  # a local shop can be a top favourite
    return pool, zipf_weights(len(pool))




GENERATOR_VERSION = "2.0"

INCOME_TYPES = ['salaried', 'gig', 'merchant', 'dependent', 'pensioner']
INCOME_TYPE_P = {          # order = INCOME_TYPES
    '18-25': [0.35, 0.15, 0.05, 0.45, 0.00],
    '26-35': [0.65, 0.18, 0.10, 0.07, 0.00],
    '36-45': [0.60, 0.15, 0.22, 0.03, 0.00],
    '46-55': [0.50, 0.12, 0.28, 0.02, 0.08],
    '56+':   [0.20, 0.05, 0.20, 0.00, 0.55],
}
AFFLUENCE_MU = {'salaried': 0.0, 'gig': -0.1, 'merchant': 0.05, 'dependent': -0.5, 'pensioner': -0.2}
AFFLUENCE_SIGMA, AFFLUENCE_CLIP = 0.30, (0.5, 2.0)
RHO_MEAN, RHO_SD, RHO_CLIP = 0.85, 0.12, (0.55, 1.20)     # total outflow / income; >1 = structurally stretched
HIDDEN_SHARE_RANGE = (0.05, 0.30)                          # h: monthly hidden ATM outflow = h x mean UPI outflow
OPENING_MULT_RANGE = (0.2, 1.5)
OPENING_MULT_RANGE_DEPENDENT = (0.05, 0.6)
TECH_FAILURE_RANGE = (0.005, 0.02)
ATM_EVENTS_PER_MONTH_RANGE = (2.0, 4.0)
NETWORK_TYPES, NETWORK_P = ['4G', '5G', 'WiFi', '3G'], [0.60, 0.25, 0.10, 0.05]
NETWORK_MULT = {'5G': 0.8, 'WiFi': 0.8, '4G': 1.0, '3G': 2.5}   # multiplies the technical-failure rate

UPI_SHARE = {'salaried': 0.0, 'pensioner': 0.0, 'dependent': 0.6, 'gig': 0.5, 'merchant': 0.9}
PAY_DAY_EARLY_P = 0.55                 # salaried: P(pay day in 1-7) else 25-31
SALARY_DELAY_P, SALARY_MONTH_SD = 0.08, 0.01
INCREMENT_P, INCREMENT_RANGE = 0.5, (0.05, 0.15)
BUMP_P, BUMP_RANGE = 0.3, (0.10, 0.40)
PENSION_DELAY_P = 0.02
GIG_CREDITS_LAMBDA_RANGE, GIG_MONTH_SD, GIG_DRY_P, GIG_DRY_FACTOR = (6, 14), 0.30, 0.15, 0.3
GIG_AMOUNT_SIGMA, GIG_ONEOFF_P = 0.8, 0.2
MERCHANT_ACTIVE_DAY_P = {'weekday': 0.88, 'weekend': 0.65}   # averages ~80% of days
MERCHANT_MONTH_SD, MERCHANT_AMOUNT_SIGMA, MERCHANT_ONEOFF_P = 0.12, 0.5, 0.3
DEPENDENT_MONTH_SD = 0.05
MAX_CREDIT_ROW_RATIO = 1.5             # credit rows must stay below this x debit attempts
INCOME_CREDIT_CAP = 1.2                # income-credit budget (x planned debits); rest is room for reciprocal/refunds

P_RENT = {'18-25': 0.25, '26-35': 0.45, '36-45': 0.30, '46-55': 0.15, '56+': 0.05}
RENT_MEDIAN, RENT_SIGMA = 9000, 0.4

P_SHOCK = 0.35
RESILIENCE_THRESHOLD, RESILIENCE_CUT = 0.5, 0.40
DISCRETIONARY = ['Food', 'Entertainment', 'Shopping']

RECIPROCAL_P, RECIPROCAL_RATIO = 0.25, (0.3, 1.2)
REFUND_P = {'Shopping': 0.06, 'Food': 0.03, 'Transport': 0.03, 'Entertainment': 0.03}
REFUND_FULL_P, REFUND_PARTIAL_RANGE = 0.7, (0.3, 0.9)

MAX_RETRIES = 3
RETRY_A = {'TECHNICAL': 0.9, 'INSUFFICIENT_FUNDS': 0.7}    # recurring / rent / Bill Payment / Recharge; wait 1-3 days
RETRY_B = {'TECHNICAL': 0.6, 'INSUFFICIENT_FUNDS': 0.3}    # organic P2P / P2M; TECHNICAL: 1 min-1 day, ISF: 1-7 days

ALLOWED_OVERRIDES = ('income_type', 'spend_ratio', 'affluence', 'opening_balance_multiple', 'pay_day',
                     'tech_failure_rate', 'shock', 'shock_month', 'shock_severity', 'resilience', 'has_rent',
                     'activity_decay', 'recurring_jitter_sd')
SHOCK_TYPES = ('none', 'income_disruption', 'expense_spike', 'both', 'random')
EVENT_PRIORITY = {'hidden_credit': 0, 'upi_credit': 1, 'hidden_debit': 2, 'upi_debit': 3}   # ties at equal timestamps
OUTPUT_COLUMNS = ['Transaction_ID', 'Timestamp', 'Direction', 'Transaction_Type', 'Merchant_Category',
                  'Counterparty_Name', 'Amount', 'Payment_Status', 'Failure_Reason', 'Balance_After',
                  'Counterparty_Bank', 'Customer_Bank', 'Device_Type', 'Network_Type', 'Hour_of_Day',
                  'Day_of_Week', 'Is_Weekend']


def _check_overrides(overrides):
    bad = [k for k in overrides if k not in ALLOWED_OVERRIDES]
    if bad:
        raise ValueError(f"Unknown override key(s) {bad}; allowed: {list(ALLOWED_OVERRIDES)}")
    if 'income_type' in overrides and overrides['income_type'] not in INCOME_TYPES:
        raise ValueError(f"income_type must be one of {INCOME_TYPES}")
    if 'shock' in overrides and overrides['shock'] not in SHOCK_TYPES:
        raise ValueError(f"shock must be one of {list(SHOCK_TYPES)}")


def build_user_profile(age_group, overrides=None):
    '''Draw the user's persistent attributes once: home state, bank, device, recurring relationships,
    favourite merchants per category, the P2P contact circle and (v2) the latent financial-health parameters.
    Every random draw is made even when the matching override is set (the value is then replaced), so the RNG
    stream stays aligned across scenario presets.'''
    overrides = dict(overrides or {})
    _check_overrides(overrides)
    probs = recurring_probability_by_age[age_group]
    services = [s for s, p in probs.items() if s not in TELECOM_OPERATORS and np.random.random() < p]

    tel_p = [probs[s] for s in TELECOM_OPERATORS]
    pick = np.random.choice(len(TELECOM_OPERATORS) + 1, p=tel_p + [1 - sum(tel_p)])
    if pick < len(TELECOM_OPERATORS):
        services.append(TELECOM_OPERATORS[pick])

    state = str(np.random.choice(STATES, p=STATE_P))

    # Each relationship gets a fixed plan price (or a base bill for varying utilities) that persists for the user
    plan_amount, receiver_name = {}, {}
    for s in services:
        plans = RECURRING_SERVICES[s][2]
        if plans is not None:
            plan_amount[s] = int(np.random.choice(plans))
        elif s == 'Electricity':
            plan_amount[s] = float(np.clip(get_amt(['Utilities'], age_group)[0], 400, 6000))
        else:  # Gas
            plan_amount[s] = float(np.random.uniform(800, 1100))
        receiver_name[s] = s
    if 'Electricity' in services:
        receiver_name['Electricity'] = str(np.random.choice(ELECTRICITY_PROVIDERS[state]))
    if 'Gas' in services:
        receiver_name['Gas'] = str(np.random.choice(GAS_PROVIDERS.get(state, DEFAULT_GAS_PROVIDERS)))
    if 'Internet' in services:
        receiver_name['Internet'] = str(np.random.choice(INTERNET_PROVIDERS))

    # Operator for one-off recharges: the recurring operator if there is one, otherwise the user's usual operator
    recurring_operator = next((s for s in services if s in TELECOM_OPERATORS), None)
    recharge_operator = recurring_operator or str(np.random.choice(TELECOM_OPERATORS, p=[0.30, 0.40, 0.20, 0.10]))

    # Favourite merchants per category (organic, non-recurring payments)
    recurring_names = list(receiver_name.values())
    pools = {c: build_receiver_pool(c, exclude=recurring_names) for c in P2M_CATEGORIES + ['Utilities']}

    # P2P: a persistent circle of contacts + a share of one-off receivers (small shops accepting on personal IDs, etc.)
    circle, circle_size = [], int(np.random.randint(8, 21))
    while len(circle) < circle_size:
        name = random_person_name()
        if name not in circle:
            circle.append(name)

    profile = {'age_group': age_group,
               'sender_state': state,
               'customer_bank': str(np.random.choice(BANKS, p=BANK_P)),
               'device_type': str(np.random.choice(['Android', 'iOS', 'Web'], p=[0.75, 0.20, 0.05])),
               'recurring_services': services,
               'recurring_operator': recurring_operator,
               'recharge_operator': recharge_operator,
               'plan_amount': plan_amount,
               'receiver_name': receiver_name,
               'pools': pools,
               'circle': circle,
               'circle_weights': zipf_weights(len(circle), 0.7, 1.2),
               'circle_share': float(np.random.uniform(0.60, 0.90))}

    # ---- v2 latent parameters (in-memory only; never emitted) ----
    income_type = INCOME_TYPES[int(np.random.choice(len(INCOME_TYPES), p=INCOME_TYPE_P[age_group]))]
    aff_z = np.random.normal()
    rho = float(np.clip(np.random.normal(RHO_MEAN, RHO_SD), *RHO_CLIP))
    h = float(np.random.uniform(*HIDDEN_SHARE_RANGE))
    open_u = float(np.random.uniform())
    tech = float(np.random.uniform(*TECH_FAILURE_RANGE))
    resilience = float(np.random.uniform())
    atm_lambda = float(np.random.uniform(*ATM_EVENTS_PER_MONTH_RANGE))
    u_pay, pay_early, pay_late = float(np.random.uniform()), int(np.random.randint(1, 8)), int(np.random.randint(25, 32))
    pay_pen, pay_dep = int(np.random.randint(1, 6)), int(np.random.randint(1, 11))
    u_rent, rent_z, rent_day = float(np.random.uniform()), np.random.normal(), int(np.random.randint(1, 6))
    landlord = random_person_name()
    while landlord in circle:
        landlord = random_person_name()
    u_shock, u_shock_type = float(np.random.uniform()), float(np.random.uniform())
    shock_month, shock_day = int(np.random.randint(4, 11)), int(np.random.randint(0, 28))
    shock_severity = float(np.random.uniform(0.0, 0.6))
    shock_months, spike_mult, spike_n = int(np.random.randint(1, 4)), float(np.random.uniform(0.4, 1.2)), int(np.random.randint(3, 7))
    u_inc, inc_pct, inc_month = float(np.random.uniform()), float(np.random.uniform(*INCREMENT_RANGE)), int(np.random.randint(1, 13))
    u_bump, bump_pct = float(np.random.uniform()), float(np.random.uniform(*BUMP_RANGE))
    bump_n, bump_m1, bump_m2 = int(np.random.randint(1, 3)), int(np.random.randint(1, 13)), int(np.random.randint(1, 13))
    gig_lambda = float(np.random.uniform(*GIG_CREDITS_LAMBDA_RANGE))
    n_clients = int(np.random.randint(5, 11))
    client_names = [random_person_name() for _ in range(10)]
    regulars = [random_person_name() for _ in range(15)]

    # ---- apply overrides (draws above are already consumed) ----
    income_type = overrides.get('income_type', income_type)
    affluence = float(np.clip(np.exp(AFFLUENCE_MU[income_type] + AFFLUENCE_SIGMA * aff_z), *AFFLUENCE_CLIP))
    affluence = float(overrides.get('affluence', affluence))
    rho = float(overrides.get('spend_ratio', rho))
    lo, hi = OPENING_MULT_RANGE_DEPENDENT if income_type == 'dependent' else OPENING_MULT_RANGE
    opening_mult = float(overrides.get('opening_balance_multiple', lo + open_u * (hi - lo)))
    tech = float(overrides.get('tech_failure_rate', tech))
    resilience = float(overrides.get('resilience', resilience))
    pay_day = {'pensioner': pay_pen, 'dependent': pay_dep}.get(income_type, pay_early if u_pay < PAY_DAY_EARLY_P else pay_late)
    pay_day = int(overrides.get('pay_day', pay_day))

    p_rent = 0.0 if income_type in ('dependent', 'pensioner') else P_RENT[age_group]
    has_rent = bool(overrides.get('has_rent', u_rent < p_rent))
    rent_amount = max(500, int(round(RENT_MEDIAN * np.exp(RENT_SIGMA * rent_z) * np.sqrt(affluence) / 500.0)) * 500)

    if u_shock >= P_SHOCK:
        shock_type = 'none'
    else:
        shock_type = 'income_disruption' if u_shock_type < 0.5 else 'expense_spike'
    if overrides.get('shock', 'random') != 'random':
        shock_type = overrides['shock']
    shock = {'type': shock_type, 'month': int(overrides.get('shock_month', shock_month)), 'day': shock_day,
             'severity': float(overrides.get('shock_severity', shock_severity)), 'months': shock_months,
             'spike_mult': spike_mult, 'n_extra': spike_n}

    profile.update({
        'income_type': income_type, 'affluence': affluence, 'rho': rho, 'h': h,
        'opening_balance_multiple': opening_mult, 'tech_failure_rate': tech, 'resilience': resilience,
        'atm_lambda': atm_lambda, 'pay_day': pay_day, 'upi_share': UPI_SHARE[income_type],
        'has_rent': has_rent, 'rent_amount': rent_amount, 'rent_day': rent_day, 'landlord_name': landlord,
        'shock': shock,
        'increment': {'on': u_inc < INCREMENT_P, 'pct': inc_pct, 'month': inc_month},
        'bump': {'on': u_bump < BUMP_P, 'pct': bump_pct, 'months': sorted({bump_m1, bump_m2} if bump_n == 2 else {bump_m1})},
        'gig_lambda': gig_lambda,
        'clients': list(dict.fromkeys(client_names))[:n_clients], 'client_weights': zipf_weights(min(n_clients, len(set(client_names))), 0.7, 1.2),
        'regulars': list(dict.fromkeys(regulars)), 'regular_weights': zipf_weights(len(set(regulars)), 0.7, 1.2),
        'activity_decay': float(overrides.get('activity_decay', 0.0)),
        'recurring_jitter_sd': float(overrides.get('recurring_jitter_sd', 1.5)),
        'generator_version': GENERATOR_VERSION})
    return profile


def random_time_of_day():
    '''Same hour/minute/second logic as generate_upi_set'''
    return timedelta(hours=int(np.random.choice(range(24), p=hourly_distribution())),
                     minutes=int(np.random.randint(0, 60)),
                     seconds=int(np.random.randint(0, 60)))


def _scale_amount(amount, affluence):
    '''Affluence scaling is applied AFTER get_amt so get_amt itself stays untouched.'''
    return int(np.clip(round(amount * affluence), 10, 100000))


def _month_starts(start, end):
    cur, out = pd.Timestamp(start).replace(day=1), []
    while cur <= end:
        out.append(cur)
        cur += pd.DateOffset(months=1)
    return out


def generate_recurring_transactions(profile, start, end):
    '''PLANNED attempts for every recurring relationship (incl. rent). Status and retries are decided by the ledger
    replay, not here. Chain = one billing cycle (a chain of attempts if the first one fails).'''
    cols = ['Timestamp', 'Transaction_Type', 'Merchant_Category', 'Counterparty_Name', 'Amount', 'Chain']
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sd = profile['recurring_jitter_sd']
    clip = max(3, int(np.ceil(2 * sd)))

    def due_date(nominal):
        return nominal + timedelta(days=int(np.clip(round(np.random.normal(0, sd)), -clip, clip)))

    rows = []
    for svc in profile['recurring_services']:
        category, ttype, plans, period = RECURRING_SERVICES[svc]
        base = profile['plan_amount'][svc]
        anchor = start + timedelta(days=int(np.random.randint(0, 28)))
        k = 0
        while True:
            nominal = anchor + (pd.DateOffset(months=k) if period == 'monthly' else timedelta(days=28 * k))
            if nominal > end:
                break
            k += 1
            due = due_date(nominal)
            if not (start <= due <= end):
                continue
            # varying bills scale with sqrt(affluence); fixed-price plans are never scaled
            amount = int(round(base if plans is not None else base * np.sqrt(profile['affluence']) * np.random.lognormal(0, 0.15)))
            rows.append([due.normalize() + random_time_of_day(), ttype, category, profile['receiver_name'][svc], amount, f'{svc}-{k}'])

    if profile['has_rent']:
        for k, ms in enumerate(_month_starts(start, end), 1):
            day = min(profile['rent_day'], ms.days_in_month)
            due = due_date(ms + timedelta(days=day - 1))
            if start <= due <= end:
                rows.append([due.normalize() + random_time_of_day(), 'P2P', 'P2P', profile['landlord_name'],
                             profile['rent_amount'], f'RENT-{k}'])
    return pd.DataFrame(rows, columns=cols)


def _organic_day_weights(profile, start, n_days):
    '''Uniform days, except pay-cycle customers (salaried/pensioner/dependent) spend a bit more right after pay day:
    weight 1 + 0.5*exp(-k/7), k = days since the most recent pay day.'''
    if profile['income_type'] not in ('salaried', 'pensioner', 'dependent'):
        return np.full(n_days, 1.0 / n_days)
    w = np.empty(n_days)
    for i, d in enumerate(pd.date_range(start, periods=n_days)):
        pay = min(profile['pay_day'], d.days_in_month)
        if d.day >= pay:
            k = d.day - pay
        else:
            prev_len = (d.replace(day=1) - timedelta(days=1)).day
            k = d.day + prev_len - min(profile['pay_day'], prev_len)
        w[i] = 1 + 0.5 * np.exp(-k / 7)
    return w / w.sum()


def _new_debit(ts, ttype, cat, cpty, amount, recurring, chain_id):
    return {'kind': 'upi_debit', 'ts': ts, 'ttype': ttype, 'cat': cat, 'cpty': cpty, 'amt': int(amount),
            'recurring': recurring, 'chain_id': chain_id, 'retry_no': 0,
            'network': str(np.random.choice(NETWORK_TYPES, p=NETWORK_P))}


def plan_debits(profile, n_records, start, end):
    '''Step 2: recurring (incl. rent) + organic debit ATTEMPTS as event dicts. Returns (events, n_recurring).'''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    date_range = (end - start).days + 1
    aff = profile['affluence']
    rec = generate_recurring_transactions(profile, start, end)
    events = [_new_debit(r.Timestamp, r.Transaction_Type, r.Merchant_Category, r.Counterparty_Name, r.Amount, True, i)
              for i, r in enumerate(rec.itertuples(index=False))]

    n_org = max(n_records - len(rec), 0)
    type_p = np.array([0.45, 0.35, 0.15, 0.05])
    if profile['recurring_operator']:          # the recurring plan already covers this user's recharges
        type_p[3] = 0.0
        type_p /= type_p.sum()
    ttype = np.random.choice(['P2P', 'P2M', 'Bill Payment', 'Recharge'], size=n_org, p=type_p)

    # Category is conditional on transaction type
    w = P2M_BASE_WEIGHTS * np.array([age_category_weight_adjustment[profile['age_group']].get(c, 1.0) for c in P2M_CATEGORIES])
    w = w / w.sum()
    category = np.where(ttype == 'P2P', 'P2P',
               np.where(ttype == 'Bill Payment', 'Utilities',
               np.where(ttype == 'Recharge', 'Telecom',
                        np.random.choice(P2M_CATEGORIES, size=n_org, p=w))))
    days = np.random.choice(date_range, size=n_org, p=_organic_day_weights(profile, start, date_range))

    for i, (t, c) in enumerate(zip(ttype, category)):
        if t == 'P2P':
            if np.random.random() < profile['circle_share']:
                cpty = str(np.random.choice(profile['circle'], p=profile['circle_weights']))
            else:
                cpty = random_person_name()
                while cpty == profile['landlord_name']:     # the landlord is only ever paid through rent
                    cpty = random_person_name()
            amount = _scale_amount(get_amt(['Other'], profile['age_group'])[0], aff)   # get_amt has no P2P branch
        elif t == 'Recharge':
            cpty = profile['recharge_operator']
            amount = int(np.random.choice(RECURRING_SERVICES[cpty][2]))                # fixed-price: not scaled
        else:
            names, p = profile['pools'][c]
            cpty = str(np.random.choice(names, p=p))
            amount = _scale_amount(get_amt([c], profile['age_group'])[0], aff)
        ts = start + timedelta(days=int(days[i])) + random_time_of_day()
        events.append(_new_debit(ts, str(t), str(c), cpty, amount, False, len(rec) + i))
    return events, len(rec)


def _thin(events, keep_prob_fn):
    return [e for e in events if e['recurring'] or np.random.random() < keep_prob_fn(e)]


def _income_factor(profile, ts):
    s = profile['shock']
    if s['type'] in ('income_disruption', 'both') and profile['income_type'] != 'pensioner' and s['start'] <= ts < s['end']:
        return s['severity']
    return 1.0


def _credit(ts, amt, ttype, cat, cpty, hidden=False):
    return {'kind': 'hidden_credit' if hidden else 'upi_credit', 'ts': ts, 'amt': round(float(amt), 2) if hidden else max(1, int(round(amt))),
            'ttype': ttype, 'cat': cat, 'cpty': cpty}


def _merge_credits(credits, max_n):
    '''Merge into larger credits when a month would exceed its row budget (contiguous chunks summed).'''
    if len(credits) <= max_n:
        return credits
    credits = sorted(credits, key=lambda e: e['ts'])
    out = []
    for chunk in np.array_split(np.arange(len(credits)), max_n):
        first = dict(credits[chunk[0]])
        first['amt'] = sum(credits[i]['amt'] for i in chunk)
        out.append(first)
    return out


def _split_month(total, k, sigma):
    w = np.random.lognormal(0, sigma, size=k)
    return total * w / w.sum()


def _pick(names, weights, oneoff_p):
    return str(np.random.choice(names, p=weights)) if np.random.random() >= oneoff_p else random_person_name()


def build_income_events(profile, start, end, monthly_income, n_planned):
    '''Step 5a: hidden bank credits + emitted UPI credits by income type (see SYNTHETIC ASSUMPTIONS).
    Emitted credits per month ~ upi_share x income; the remainder is hidden (visible only as balance jumps).'''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    end_excl = end + timedelta(days=1)
    it, share = profile['income_type'], profile['upi_share']
    months = _month_starts(start, end)
    cap_m = max(1, int(INCOME_CREDIT_CAP * n_planned / max(len(months), 1)))
    events = []

    def hid(ts, amt):
        events.append(_credit(ts, amt * _income_factor(profile, ts), None, None, None, hidden=True))

    def emit(ts, amt, ttype, cat, cpty):
        return _credit(ts, amt * _income_factor(profile, ts), ttype, cat, cpty)

    def morning():
        return timedelta(hours=int(np.random.randint(6, 12)), minutes=int(np.random.randint(0, 60)))

    for mi, ms in enumerate(months, 1):
        dim = ms.days_in_month
        emitted = []
        if it in ('salaried', 'pensioner'):
            day = min(profile['pay_day'], dim)
            ts = ms + timedelta(days=day - 1)
            if it == 'salaried':
                if ts.weekday() >= 5:                                   # weekend -> preceding Friday
                    ts -= timedelta(days=ts.weekday() - 4)
                if np.random.random() < SALARY_DELAY_P:
                    ts += timedelta(days=int(np.random.randint(1, 4)))
                amt = monthly_income * np.random.normal(1, SALARY_MONTH_SD)
                if profile['increment']['on'] and mi >= profile['increment']['month']:
                    amt *= 1 + profile['increment']['pct']
                if profile['bump']['on'] and mi in profile['bump']['months']:
                    amt *= 1 + profile['bump']['pct']
            else:
                if np.random.random() < PENSION_DELAY_P:
                    ts += timedelta(days=int(np.random.randint(1, 4)))
                amt = monthly_income
            hid(ts + morning(), amt)
        elif it == 'dependent':
            inc = monthly_income * np.random.normal(1, DEPENDENT_MONTH_SD)
            k = int(np.random.randint(1, 4))
            parts = [(0.75, int(np.clip(profile['pay_day'] + np.random.randint(-1, 2), 1, dim)))]
            parts += [(0.25 / k, int(np.random.randint(1, dim + 1))) for _ in range(k)]
            for frac, day in parts:
                ts = ms + timedelta(days=day - 1) + random_time_of_day()
                payer = str(np.random.choice(profile['circle'][:3]))
                emitted.append(emit(ts, inc * frac * share, 'P2P', 'P2P', payer))
                hid(ts, inc * frac * (1 - share))
        elif it == 'gig':
            factor = GIG_DRY_FACTOR if np.random.random() < GIG_DRY_P else max(0.2, np.random.normal(1, GIG_MONTH_SD))
            inc = monthly_income * factor
            k = max(1, int(np.random.poisson(profile['gig_lambda'])))
            for amt in _split_month(inc * share, k, GIG_AMOUNT_SIGMA):
                ts = ms + timedelta(days=int(np.random.randint(0, dim))) + random_time_of_day()
                cpty = _pick(profile['clients'], profile['client_weights'], GIG_ONEOFF_P)
                is_client = cpty in profile['clients']
                emitted.append(emit(ts, amt, 'P2P' if is_client else 'P2M', 'P2P' if is_client else 'Collection', cpty))
            weekdays = list(range(int(np.random.randint(0, 7)), dim, 7))    # weekly lumpy payouts, remainder hidden
            for amt, d in zip(_split_month(inc * (1 - share), len(weekdays), 0.5), weekdays):
                hid(ms + timedelta(days=d) + morning(), amt)
        else:  # merchant
            inc = monthly_income * np.random.normal(1, MERCHANT_MONTH_SD)
            slots = []
            for d in range(dim):
                p_active = MERCHANT_ACTIVE_DAY_P['weekend' if (ms + timedelta(days=d)).weekday() >= 5 else 'weekday']
                if np.random.random() < p_active:
                    slots += [d] * int(np.random.randint(1, 4))
            for amt, d in zip(_split_month(inc * share, max(len(slots), 1), MERCHANT_AMOUNT_SIGMA), slots or [0]):
                ts = ms + timedelta(days=d) + random_time_of_day()
                emitted.append(emit(ts, amt, 'P2M', 'Collection', _pick(profile['regulars'], profile['regular_weights'], MERCHANT_ONEOFF_P)))
            hid(ms + timedelta(days=min(27, dim - 1)) + morning(), inc * (1 - share))
        events += _merge_credits(emitted, cap_m)
    return [e for e in events if start <= e['ts'] < end_excl]


def build_hidden_outflows(profile, start, end, monthly_outflow, monthly_income):
    '''Step 5b: hidden ATM cash (multiples of 500; monthly total ~ h x mean UPI outflow) and the shock one-off bank debit.'''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    end_excl = end + timedelta(days=1)
    events = []
    for ms in _month_starts(start, end):
        n = max(1, int(np.random.poisson(profile['atm_lambda'])))
        for wi in np.random.dirichlet(np.ones(n)):
            amt = max(500, int(round(profile['h'] * monthly_outflow * wi / 500.0)) * 500)
            ts = ms + timedelta(days=int(np.random.randint(0, ms.days_in_month))) + random_time_of_day()
            events.append({'kind': 'hidden_debit', 'ts': ts, 'amt': float(amt)})
    s = profile['shock']
    if s['type'] in ('expense_spike', 'both'):
        events.append({'kind': 'hidden_debit', 'ts': s['start'] + timedelta(hours=9),
                       'amt': round(s['spike_mult'] * monthly_income, 2)})
    return [e for e in events if start <= e['ts'] < end_excl]


def replay_ledger(events, profile, opening_balance, end_excl):
    '''Step 6: heap-ordered replay of every event against a single balance. Returns (rows, hidden_applied, chains).
    Balance never goes negative; failed debits leave it unchanged; retries are new attempts pushed into the queue.'''
    heap, counter = [], itertools.count()

    def push(ev):
        heapq.heappush(heap, (ev['ts'], EVENT_PRIORITY[ev['kind']], next(counter), ev))

    for ev in events:
        push(ev)
    bal = round(float(opening_balance), 2)
    rows, hidden, chains = [], [], {}
    circle_set = set(profile['circle'])

    def row(ts, ev, direction, status, reason, balance):
        debit = direction == 'DEBIT'
        rows.append({'Timestamp': ts, 'Direction': direction, 'Transaction_Type': ev['ttype'],
                     'Merchant_Category': ev['cat'], 'Counterparty_Name': ev['cpty'], 'Amount': ev['amt'],
                     'Payment_Status': status, 'Failure_Reason': reason, 'Balance_After': round(balance, 2),
                     'Device_Type': profile['device_type'] if debit else None,
                     'Network_Type': ev['network'] if debit else None})

    while heap:
        ts, _, _, ev = heapq.heappop(heap)
        kind = ev['kind']
        if kind == 'hidden_credit':
            bal = round(bal + ev['amt'], 2)
            hidden.append({'ts': ts, 'amount': ev['amt'], 'after_row': len(rows)})
        elif kind == 'hidden_debit':
            applied = round(min(ev['amt'], bal), 2)
            bal = round(bal - applied, 2)
            hidden.append({'ts': ts, 'amount': -applied, 'after_row': len(rows)})
        elif kind == 'upi_credit':
            bal = round(bal + ev['amt'], 2)
            row(ts, ev, 'CREDIT', 'SUCCESS', None, bal)
        else:
            amt = ev['amt']
            u = np.random.random()
            if amt > bal:
                status, reason = 'FAILED', 'INSUFFICIENT_FUNDS'
            elif u < profile['tech_failure_rate'] * NETWORK_MULT[ev['network']]:
                status, reason = 'FAILED', 'TECHNICAL'
            else:
                status, reason = 'SUCCESS', None
                bal = round(bal - amt, 2)
            row(ts, ev, 'DEBIT', status, reason, bal)
            ch = chains.setdefault(ev['chain_id'], {'attempts': 0, 'outcome': None})
            ch['attempts'] += 1
            ch['outcome'] = status

            if status == 'FAILED' and ev['retry_no'] < MAX_RETRIES:
                class_a = ev['recurring'] or ev['ttype'] in ('Bill Payment', 'Recharge')
                p = (RETRY_A if class_a else RETRY_B)[reason]
                if np.random.random() < p:
                    if class_a:
                        nts = ts.normalize() + timedelta(days=int(np.random.randint(1, 4))) + random_time_of_day()
                    elif reason == 'TECHNICAL':
                        nts = ts + timedelta(seconds=int(np.random.randint(60, 86401)))
                    else:
                        nts = ts.normalize() + timedelta(days=int(np.random.randint(1, 8))) + random_time_of_day()
                    if nts < end_excl:
                        retry = dict(ev, ts=nts, retry_no=ev['retry_no'] + 1,
                                     network=str(np.random.choice(NETWORK_TYPES, p=NETWORK_P)))
                        push(retry)
            elif status == 'SUCCESS' and not ev['recurring']:
                if ev['ttype'] == 'P2P' and ev['cpty'] in circle_set and np.random.random() < RECIPROCAL_P:
                    nts = ts.normalize() + timedelta(days=int(np.random.randint(1, 21))) + random_time_of_day()
                    payer = str(np.random.choice(profile['circle'], p=profile['circle_weights']))
                    amt_back = amt * np.random.uniform(*RECIPROCAL_RATIO)
                    if nts < end_excl:
                        push(_credit(nts, amt_back, 'P2P', 'P2P', payer))
                elif ev['ttype'] == 'P2M' and np.random.random() < REFUND_P.get(ev['cat'], 0.0):
                    full = np.random.random() < REFUND_FULL_P
                    amt_back = amt if full else amt * np.random.uniform(*REFUND_PARTIAL_RANGE)
                    nts = ts.normalize() + timedelta(days=int(np.random.randint(2, 8))) + random_time_of_day()
                    if nts < end_excl:
                        push(_credit(nts, amt_back, 'Refund', ev['cat'], ev['cpty']))
    return rows, hidden, chains


def generate_user_transactions(age_group='26-35', n_records=600, start_date='2024-01-01', end_date='2024-12-31',
                               seed=1, n_variation=0.10, overrides=None, return_profile=False):
    '''UPI history (debits AND credits) for ONE persistent customer with a hidden bank-account ledger.
    n_records = target number of planned DEBIT ATTEMPTS (organic + recurring + rent), drawn uniformly within
    +/- n_variation. Credits, retry attempts and shock-injected extra debits are additional, so total rows > n_records.
    overrides: dict of latent-parameter overrides (keys in ALLOWED_OVERRIDES; unknown key -> ValueError).
    Returns the DataFrame, or (DataFrame, profile) if return_profile=True; the profile holds latent parameters and
    hidden events for validation only and must never be written next to the CSV.'''
    overrides = dict(overrides or {})
    _check_overrides(overrides)
    np.random.seed(seed)
    profile = build_user_profile(age_group, overrides)
    n_target = int(round(n_records * np.random.uniform(1 - n_variation, 1 + n_variation)))

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    end_excl = end + timedelta(days=1)
    n_days = (end - start).days + 1

    shock = profile['shock']
    shock['start'] = min(start + pd.DateOffset(months=shock['month'] - 1) + timedelta(days=shock['day']), end)
    shock['end'] = shock['start'] + pd.DateOffset(months=shock['months'])

    # 2-3. plan debit attempts, then activity-decay thinning (organic only)
    debits, n_rec = plan_debits(profile, n_target, start, end)
    decay = profile['activity_decay']
    if decay > 0:
        debits = _thin(debits, lambda e: 1 - decay * (e['ts'] - start) / (end_excl - start))
    n_planned = len(debits)

    # 4. size income FROM planned outflow (synthetic-data convention, not causal realism)
    monthly_outflow = sum(e['amt'] for e in debits) / (n_days / 30.4375)
    monthly_income = max(monthly_outflow * (1 + profile['h']) / profile['rho'], 1.0)
    opening_balance = round(profile['opening_balance_multiple'] * monthly_income, 2)

    # 5. income, hidden outflows, shock effects
    events = build_income_events(profile, start, end, monthly_income, n_planned)
    events += build_hidden_outflows(profile, start, end, monthly_outflow, monthly_income)
    if shock['type'] != 'none' and profile['resilience'] >= RESILIENCE_THRESHOLD:
        w_end = shock['end'] + pd.DateOffset(months=1)
        debits = [e for e in debits if e['recurring'] or e['cat'] not in DISCRETIONARY or not (shock['start'] <= e['ts'] < w_end)
                  or np.random.random() >= RESILIENCE_CUT]
    if shock['type'] in ('expense_spike', 'both'):
        names, p = profile['pools']['Healthcare']
        for _ in range(shock['n_extra']):
            ts = shock['start'] + timedelta(days=int(np.random.randint(0, 14))) + random_time_of_day()
            amount = _scale_amount(get_amt(['Healthcare'], age_group)[0], profile['affluence'])
            if ts < end_excl:
                debits.append(_new_debit(ts, 'P2M', 'Healthcare', str(np.random.choice(names, p=p)), amount, False, 10**6 + len(debits)))

    # 6. replay
    rows, hidden, chains = replay_ledger(events + debits, profile, opening_balance, end_excl)

    df = pd.DataFrame(rows)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.insert(0, 'Transaction_ID', [f'TXN{str(i).zfill(10)}' for i in range(1, len(df) + 1)])
    bank_of = {}  # a counterparty keeps the same bank across the whole history
    for name in df['Counterparty_Name']:
        if name not in bank_of:
            bank_of[name] = str(np.random.choice(BANKS, p=BANK_P))
    df['Counterparty_Bank'] = df['Counterparty_Name'].map(bank_of)
    df['Customer_Bank'] = profile['customer_bank']
    df['Hour_of_Day'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.day_name()
    df['Is_Weekend'] = (df['Timestamp'].dt.weekday >= 5).astype(int)
    df = df[OUTPUT_COLUMNS]
    df.attrs['generator_version'] = GENERATOR_VERSION

    if return_profile:
        profile.update({'n_records_target': n_target, 'planned_debit_attempts': n_planned, 'n_recurring_planned': n_rec,
                        'monthly_outflow': monthly_outflow, 'monthly_income': monthly_income,
                        'opening_balance': opening_balance, 'hidden_events': hidden,
                        'retry_chains': [c for c in chains.values() if c['attempts'] > 1],
                        'n_chains': len(chains)})
        return df, profile
    return df


# ---------------- Inference helpers (use only the emitted DataFrame) ----------------
def detect_recurring(d):
    '''Recurring relationships inferred purely from history (DEBIT rows only): same counterparty, successful payments in
    >=6 distinct months, roughly evenly spaced ~4-5 weeks apart. Rent = P2P, gaps 24-35 days, gap std < 8,
    amount CV < 0.05.'''
    out = {}
    d = d[(d['Direction'] == 'DEBIT') & (d['Payment_Status'] == 'SUCCESS')]
    ok = d[d['Merchant_Category'].isin(['Entertainment', 'Telecom', 'Utilities'])]
    for name, g in ok.groupby('Counterparty_Name'):
        ts = g['Timestamp'].sort_values()
        gaps = ts.diff().dt.days.dropna()
        if len(ts) >= 6 and ts.dt.to_period('M').nunique() >= 6 and 24 <= gaps.median() <= 35 and gaps.std() < 8:
            out[name] = g.sort_values('Timestamp')
    for name, g in d[d['Transaction_Type'] == 'P2P'].groupby('Counterparty_Name'):
        ts = g['Timestamp'].sort_values()
        gaps = ts.diff().dt.days.dropna()
        if (len(ts) >= 6 and ts.dt.to_period('M').nunique() >= 6 and 24 <= gaps.median() <= 35 and gaps.std() < 8
                and g['Amount'].std() / g['Amount'].mean() < 0.05):
            out[name] = g.sort_values('Timestamp')
    return out


def unexplained_flows(df, opening_balance=None):
    '''Balance change between consecutive rows not explained by the row's own successful UPI amount = net hidden flow.
    The first row is only comparable if the opening balance is known.'''
    signed = np.where(df['Payment_Status'] == 'SUCCESS', np.where(df['Direction'] == 'CREDIT', 1.0, -1.0) * df['Amount'], 0.0)
    prev = np.concatenate([[np.nan if opening_balance is None else opening_balance], df['Balance_After'].to_numpy()[:-1]])
    return df['Balance_After'].to_numpy() - prev - signed


def check_reconciliation(df, profile, tol=0.01):
    '''Every balance change must equal the row's own successful UPI amount plus the hidden flows applied just before it.'''
    net = np.zeros(len(df) + 1)
    for h in profile['hidden_events']:
        net[h['after_row']] += h['amount']
    resid = unexplained_flows(df, profile['opening_balance']) - net[:len(df)]
    bad = int((np.abs(resid) > tol + 1e-6).sum())
    return bad, float(np.abs(resid).max()) if len(resid) else 0.0


def infer_pay_cycle_day(df):
    '''From the balance series ALONE: day-of-month (mode) of large unexplained positive jumps (> 50% of the median
    monthly-maximum jump). Returns None if there are no jumps.'''
    jump = unexplained_flows(df)[1:]
    sub = df.iloc[1:].assign(jump=jump)
    sub = sub[sub['jump'] > 1.0]
    if sub.empty:
        return None
    monthly_max = sub.groupby(sub['Timestamp'].dt.to_period('M'))['jump'].max()
    big = sub[sub['jump'] > 0.5 * monthly_max.median()]
    return int(big['Timestamp'].dt.day.mode().iloc[0])


def _circ_dist(a, b, period=30):
    d = abs(a - b) % period
    return min(d, period - d)


def verify_user_history(df, n_records, profile=None):
    '''Print the checks requested for the single-user history. Warns (never raises) on failed invariants.'''
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    warnings = []
    print(df.head(10).to_string(), '\n')
    deb, cre = df[df['Direction'] == 'DEBIT'], df[df['Direction'] == 'CREDIT']
    print(f"Rows: {len(df)} = {len(deb)} debit attempts + {len(cre)} credits (target {n_records} planned debit attempts "
          f"+/- 10%; extra rows come from credits, retries and shock debits)")
    print(f"Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}  "
          f"({df['Timestamp'].dt.to_period('M').nunique()} distinct months) | generator_version={df.attrs.get('generator_version')}")
    forbidden = [c for c in df.columns if c in ['Customer_ID', 'User_ID', 'isRecurring', 'Sender_Age_Group', 'Fraud_Flag',
                                                'Income', 'Salary'] or 'score' in c.lower()]
    print("Forbidden columns present:", forbidden)
    print("Constant user attributes:", df[['Customer_Bank']].nunique().to_dict(), "| Device_Type (debits):",
          deb['Device_Type'].unique().tolist(), "| Device/Network null on credits:",
          bool(cre[['Device_Type', 'Network_Type']].isna().all().all()))
    print("\nDirection x Transaction_Type:\n", pd.crosstab(df['Direction'], df['Transaction_Type']), sep='')
    print("\nMerchant categories:\n", df['Merchant_Category'].value_counts(), sep='')
    print("\nCategory x type:\n", pd.crosstab(df['Merchant_Category'], df['Transaction_Type']), sep='')
    print("\nP2P rows with category != 'P2P':", int(((df['Transaction_Type'] == 'P2P') & (df['Merchant_Category'] != 'P2P')).sum()),
          "| non-P2P rows with category 'P2P':", int(((df['Transaction_Type'] != 'P2P') & (df['Merchant_Category'] == 'P2P')).sum()))
    print("Null counterparties:", int(df['Counterparty_Name'].isna().sum()))

    per_month = df.groupby(df['Timestamp'].dt.to_period('M'))
    print("\nRows per month:", per_month.size().to_dict())
    print("Credit rows per month:", cre.groupby(cre['Timestamp'].dt.to_period('M')).size().to_dict(),
          f"| credit share {len(cre) / len(df):.1%}")

    failed = deb[deb['Payment_Status'] == 'FAILED']
    isf, tech = (failed['Failure_Reason'] == 'INSUFFICIENT_FUNDS').sum(), (failed['Failure_Reason'] == 'TECHNICAL').sum()
    share = len(failed) / max(len(deb), 1)
    print(f"\nFailed debits: {len(failed)}/{len(deb)} = {share:.2%} (INSUFFICIENT_FUNDS {isf}, TECHNICAL {tech})")
    if share > 0.15:
        warnings.append(f"failed share {share:.1%} > 15%")
    inv = []
    if (df['Balance_After'] < 0).any():
        inv.append("negative balance")
    if (cre['Payment_Status'] != 'SUCCESS').any():
        inv.append("failed credit")
    if df.loc[df['Payment_Status'] == 'SUCCESS', 'Failure_Reason'].notna().any() or df.loc[df['Payment_Status'] == 'FAILED', 'Failure_Reason'].isna().any():
        inv.append("Failure_Reason inconsistent with status")
    if (failed[failed['Failure_Reason'] == 'INSUFFICIENT_FUNDS']['Amount'] <= failed[failed['Failure_Reason'] == 'INSUFFICIENT_FUNDS']['Balance_After']).any():
        inv.append("INSUFFICIENT_FUNDS row with Amount <= balance")

    bal = df['Balance_After']
    daily = bal.groupby(df['Timestamp'].dt.normalize()).last()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).ffill()
    print(f"\nBalance: min {bal.min():,.0f}, median {bal.median():,.0f}, "
          f"share of days below 10% of median balance {(daily < 0.1 * bal.median()).mean():.1%}")

    if profile is not None:
        bad, worst = check_reconciliation(df, profile)
        print(f"Reconciliation: {bad} mismatching rows (max abs residual {worst:.4f})")
        if bad:
            inv.append(f"{bad} reconciliation mismatches")
    for msg in inv:
        warnings.append("INVARIANT FAILED: " + msg)

    multi = (profile or {}).get('retry_chains')
    if multi is not None:
        out = pd.Series([c['outcome'] for c in multi]).value_counts().to_dict()
        print(f"Retry chains (>=2 attempts): {len(multi)} of {profile['n_chains']}; final outcome {out}; "
              f"max attempts {max([c['attempts'] for c in multi], default=1)}")

    inferred = infer_pay_cycle_day(df)
    if profile is not None and profile['income_type'] in ('salaried', 'pensioner'):
        print(f"Pay-cycle recovery from balances alone: inferred day {inferred} vs profile pay_day {profile['pay_day']} "
              f"(circular distance {_circ_dist(inferred, profile['pay_day']) if inferred else 'n/a'})")
    else:
        print(f"Pay-cycle inferred from balances (largest unexplained jumps): day {inferred}")

    print("\nWeekend share:", round(df['Is_Weekend'].mean(), 3), "(uniform days would give ~0.286)")
    print("Peak hours (top 5):", df['Hour_of_Day'].value_counts().head(5).to_dict())
    print("\nMedian amount by category:\n", df.groupby('Merchant_Category')['Amount'].median(), sep='')
    p2p = deb[deb['Transaction_Type'] == 'P2P']['Counterparty_Name']
    print(f"\nP2P debit counterparties: {p2p.nunique()} distinct over {len(p2p)} payments; "
          f"top 5: {p2p.value_counts().head(5).to_dict()}")
    rec = detect_recurring(df)
    print(f"\nRecurring relationships detected: {len(rec)}"
          + (f" | rent detected: {[n for n, g in rec.items() if g['Transaction_Type'].iloc[0] == 'P2P']}"))
    for name, g in rec.items():
        print(f"  {name} ({g['Merchant_Category'].iloc[0]}): {len(g)} payments, amounts {[int(a) for a in sorted(g['Amount'].unique())[:6]]}")
        print("    " + ", ".join(g['Timestamp'].dt.strftime('%d-%b').tolist()))
    if profile is not None:
        s = profile['shock']
        print(f"\nLatent profile (NOT in CSV): income_type={profile['income_type']}, rho={profile['rho']:.2f}, "
              f"affluence={profile['affluence']:.2f}, shock={s['type']}"
              + (f" on {s['start'].date()} (severity {s['severity']:.2f}, {s['months']} mo)" if s['type'] != 'none' else ''))
    for w in warnings:
        print("WARNING:", w)


if __name__ == '__main__':
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    df_user, prof = generate_user_transactions(age_group='26-35', n_records=n_records,
                                               start_date='2024-01-01', end_date='2024-12-31', return_profile=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user001.csv')
    df_user.to_csv(out, index=False)
    print(f"Saved {out}\n")
    verify_user_history(df_user, n_records, prof)
