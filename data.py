import pandas as pd

def get_data():
    data = {
        "Naam": ["Anwar", "Fatima", "Jasper"],
        "Leeftijd": [28, 34, 25],
        "Stad": ["Lelystad", "Amsterdam", "Utrecht"]
    }
    df = pd.DataFrame(data)
    return df
