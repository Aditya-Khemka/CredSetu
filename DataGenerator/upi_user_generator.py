"""Single-user UPI transaction generator -- the single source of truth for generation logic.

Derived from upi-transactions-generator.ipynb: hourly_distribution() and get_amt() are copied verbatim from the
original Kaggle generator; everything after them is the single-user extension.

Use:  from upi_user_generator import generate_user_transactions, verify_user_history
      python driver.py   (or)   python upi_user_generator.py [n_records]   ->  writes user001.csv
"""
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


def build_user_profile(age_group):
    '''Draw the user's persistent attributes once: home state, bank, device, recurring relationships,
    favourite merchants per category and the P2P contact circle.'''
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
    recurring_names = set(receiver_name.values())
    pools = {c: build_receiver_pool(c, exclude=recurring_names) for c in P2M_CATEGORIES + ['Utilities']}

    # P2P: a persistent circle of contacts + a share of one-off receivers (small shops accepting on personal IDs, etc.)
    circle, circle_size = [], int(np.random.randint(8, 21))
    while len(circle) < circle_size:
        name = random_person_name()
        if name not in circle:
            circle.append(name)

    return {'age_group': age_group,
            'sender_state': state,
            'sender_bank': str(np.random.choice(BANKS, p=BANK_P)),
            'device_type': str(np.random.choice(['Android', 'iOS', 'Web'], p=[0.75, 0.20, 0.05])),
            'recurring_services': services,
            'recurring_operator': recurring_operator,
            'recharge_operator': recharge_operator,
            'plan_amount': plan_amount,
            'receiver_name': receiver_name,
            'pools': pools,
            'circle': circle,
            'circle_weights': zipf_weights(len(circle), 0.7, 1.2),
            'circle_share': float(np.random.uniform(0.60, 0.90)),
            'failure_rate': float(np.random.uniform(0.03, 0.07)),   # this user's share of FAILED payments
            'fraud_rate': float(np.random.uniform(0.0, 0.002))}     # this user's share of fraud-flagged payments


def random_time_of_day():
    '''Same hour/minute/second logic as generate_upi_set'''
    return timedelta(hours=int(np.random.choice(range(24), p=hourly_distribution())),
                     minutes=int(np.random.randint(0, 60)),
                     seconds=int(np.random.randint(0, 60)))


def generate_recurring_transactions(profile, start, end):
    '''Periodic transactions to the same receiver for every recurring relationship of the user.
    A failed payment is retried successfully 1-3 days later.'''
    cols = ['Timestamp', 'Transaction_Type', 'Merchant_Category', 'Receiver_Name', 'Amount', 'Payment_Status']
    rows = []
    for svc in profile['recurring_services']:
        category, ttype, plans, period = RECURRING_SERVICES[svc]
        base = profile['plan_amount'][svc]
        anchor = pd.Timestamp(start + timedelta(days=int(np.random.randint(0, 28))))
        k = 0
        while True:
            nominal = anchor + (pd.DateOffset(months=k) if period == 'monthly' else timedelta(days=28 * k))
            if nominal > end:
                break
            k += 1
            due = nominal + timedelta(days=int(np.clip(round(np.random.normal(0, 1.5)), -3, 3)))
            if not (start <= due <= end):
                continue
            amount = int(round(base if plans is not None else base * np.random.lognormal(0, 0.15)))
            fr = profile['failure_rate']
            status = str(np.random.choice(['SUCCESS', 'FAILED'], p=[1 - fr, fr]))
            row = [due.normalize() + random_time_of_day(), ttype, category, profile['receiver_name'][svc], amount]
            rows.append(row + [status])
            if status == 'FAILED':
                retry = due.normalize() + timedelta(days=int(np.random.randint(1, 4)))
                if retry <= end:
                    rows.append([retry + random_time_of_day()] + row[1:] + ['SUCCESS'])
    return pd.DataFrame(rows, columns=cols)


def generate_user_transactions(age_group='26-35', n_records=600, start_date='2024-01-01', end_date='2024-12-31',
                               seed=1, n_variation=0.10):
    '''UPI transaction history for ONE persistent user (the file/dataframe itself represents the user).
    n_records = target total transactions (recurring + organic); the actual count is drawn uniformly from
    n_records +/- n_variation (default 10%).  Return type : pandas_dataframe'''
    np.random.seed(seed)
    profile = build_user_profile(age_group)
    n_records = int(round(n_records * np.random.uniform(1 - n_variation, 1 + n_variation)))

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days + 1  # include the last day

    # 1. Recurring payments (persistent receiver relationships)
    rec = generate_recurring_transactions(profile, start, end)

    # 2. Organic transactions: original type mix, timing and amount logic
    n_org = max(n_records - len(rec), 0)
    type_p = np.array([0.45, 0.35, 0.15, 0.05])
    if profile['recurring_operator']:          # the recurring plan already covers this user's recharges
        type_p[3] = 0.0
        type_p /= type_p.sum()
    ttype = np.random.choice(['P2P', 'P2M', 'Bill Payment', 'Recharge'], size=n_org, p=type_p)

    # Category is conditional on transaction type
    w = P2M_BASE_WEIGHTS * np.array([age_category_weight_adjustment[age_group].get(c, 1.0) for c in P2M_CATEGORIES])
    w = w / w.sum()
    category = np.where(ttype == 'P2P', 'P2P',
               np.where(ttype == 'Bill Payment', 'Utilities',
               np.where(ttype == 'Recharge', 'Telecom',
                        np.random.choice(P2M_CATEGORIES, size=n_org, p=w))))

    # Receiver and amount per transaction
    receiver, amount = [], []
    for t, c in zip(ttype, category):
        if t == 'P2P':
            if np.random.random() < profile['circle_share']:
                receiver.append(str(np.random.choice(profile['circle'], p=profile['circle_weights'])))
            else:
                receiver.append(random_person_name())
            amount.append(get_amt(['Other'], age_group)[0])   # get_amt has no P2P branch -> generic 'Other' amounts
        elif t == 'Recharge':
            op = profile['recharge_operator']
            receiver.append(op)
            amount.append(int(np.random.choice(RECURRING_SERVICES[op][2])))
        else:
            names, p = profile['pools'][c]
            receiver.append(str(np.random.choice(names, p=p)))
            amount.append(get_amt([c], age_group)[0])

    org = pd.DataFrame({
        'Timestamp': [start + timedelta(days=int(np.random.randint(0, date_range))) + random_time_of_day()
                      for _ in range(n_org)],
        'Transaction_Type': ttype,
        'Merchant_Category': category,
        'Receiver_Name': receiver,
        'Amount': amount,
        'Payment_Status': np.random.choice(['SUCCESS', 'FAILED'], size=n_org,
                                          p=[1 - profile['failure_rate'], profile['failure_rate']])})

    df = pd.concat([org, rec], ignore_index=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # stays datetime even when the user has no recurring payments
    df = df.sort_values('Timestamp').reset_index(drop=True)
    n = len(df)
    df.insert(0, 'Transaction_ID', [f'TXN{str(i).zfill(10)}' for i in range(1, n + 1)])

    # 3. Remaining fields: user-level values persist, per-transaction fields use the original distributions
    receiver_bank = {}  # a receiver keeps the same bank across the whole history
    for name in df['Receiver_Name']:
        if name not in receiver_bank:
            receiver_bank[name] = str(np.random.choice(BANKS, p=BANK_P))
    df['Receiver_Bank'] = df['Receiver_Name'].map(receiver_bank)
    df['Sender_Bank'] = profile['sender_bank']
    df['Device_Type'] = profile['device_type']
    df['Network_Type'] = np.random.choice(['4G', '5G', 'WiFi', '3G'], size=n, p=[0.60, 0.25, 0.10, 0.05])
    df['Fraud_Flag'] = np.random.choice([0, 1], size=n, p=[1 - profile['fraud_rate'], profile['fraud_rate']])

    # derived fields
    df['Hour_of_Day'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.day_name()
    df['Is_Weekend'] = (df['Timestamp'].dt.weekday >= 5).astype(int)
    return df


def detect_recurring(d):
    '''Recurring relationships inferred purely from history: same receiver, successful payments in >=6 distinct
    months, roughly evenly spaced ~4-5 weeks apart.'''
    out = {}
    ok = d[(d['Payment_Status'] == 'SUCCESS') & d['Merchant_Category'].isin(['Entertainment', 'Telecom', 'Utilities'])]
    for name, g in ok.groupby('Receiver_Name'):
        ts = g['Timestamp'].sort_values()
        gaps = ts.diff().dt.days.dropna()
        if len(ts) >= 6 and ts.dt.to_period('M').nunique() >= 6 and 24 <= gaps.median() <= 35 and gaps.std() < 8:
            out[name] = g.sort_values('Timestamp')
    return out


def verify_user_history(df, n_records):
    '''Print the checks requested for the single-user history.'''
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(df.head(10).to_string(), '\n')
    print(f"Transactions: {len(df)} (target {n_records} +/- 10%)")
    print(f"Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}  "
          f"({df['Timestamp'].dt.to_period('M').nunique()} distinct months)")
    print("Forbidden columns present:", [c for c in ['Customer_ID', 'User_ID', 'isRecurring', 'Sender_Age_Group']
                                          if c in df.columns])
    print("Constant user attributes:", df[['Sender_Bank', 'Device_Type']].nunique().to_dict())
    print("\nTransaction types:\n", df['Transaction_Type'].value_counts(), sep='')
    print("\nMerchant categories:\n", df['Merchant_Category'].value_counts(), sep='')
    print("\nCategory x type:\n", pd.crosstab(df['Merchant_Category'], df['Transaction_Type']), sep='')
    print("\nP2P rows with category != 'P2P':", int(((df['Transaction_Type'] == 'P2P') & (df['Merchant_Category'] != 'P2P')).sum()),
          "| non-P2P rows with category 'P2P':", int(((df['Transaction_Type'] != 'P2P') & (df['Merchant_Category'] == 'P2P')).sum()))
    print("Null receivers:", int(df['Receiver_Name'].isna().sum()))
    print("\nPayment status:\n", df['Payment_Status'].value_counts(), sep='')
    print(f"Failed share: {(df['Payment_Status'] == 'FAILED').mean():.2%} | fraud-flagged: {int(df['Fraud_Flag'].sum())}")
    print("\nTransactions per month:\n", df.groupby(df['Timestamp'].dt.to_period('M')).size().to_dict(), sep='')
    print("Weekend share:", round(df['Is_Weekend'].mean(), 3), "(uniform days would give ~0.286)")
    print("Peak hours (top 5):", df['Hour_of_Day'].value_counts().head(5).to_dict())
    print("\nMedian amount by category:\n", df.groupby('Merchant_Category')['Amount'].median(), sep='')
    p2p = df[df['Transaction_Type'] == 'P2P']['Receiver_Name']
    print(f"\nP2P receivers: {p2p.nunique()} distinct over {len(p2p)} payments; "
          f"top 5: {p2p.value_counts().head(5).to_dict()}")
    rec = detect_recurring(df)
    print(f"\nRecurring relationships detected: {len(rec)}")
    for name, g in rec.items():
        print(f"  {name} ({g['Merchant_Category'].iloc[0]}): {len(g)} payments, amounts {[int(a) for a in sorted(g['Amount'].unique())[:6]]}")
        print("    " + ", ".join(g['Timestamp'].dt.strftime('%d-%b').tolist()))


if __name__ == '__main__':
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    df_user = generate_user_transactions(age_group='26-35', n_records=n_records,
                                         start_date='2024-01-01', end_date='2024-12-31')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user001.csv')
    df_user.to_csv(out, index=False)
    print(f"Saved {out}\n")
    verify_user_history(df_user, n_records)
