
# Finding similar company name and auto matching them

This program will use NLP and ML technique to match similar company names. Matching form common words like "LTD" and "COMPANY" will be discounted autometically in the algorithm.

Library used:
* pandas
* fuzzywuzzy

## Load the table in pandas DataFrame

The data we used is found on http://download.companieshouse.gov.uk/en_output.html it is an openly licensed publicly avalible dataset that contains a list of registered (limited liability) companies in Great Britain. *(the version shown here is snapshot of May 2018)*

```python
import pandas as pd
pd.set_option('display.max_columns', 1000)
df = pd.read_csv("BasicCompanyDataAsOneFile-2018-05-01.csv")
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CompanyName</th>
      <th>CompanyNumber</th>
      <th>RegAddress.CareOf</th>
      <th>RegAddress.POBox</th>
      <th>RegAddress.AddressLine1</th>
      <th>RegAddress.AddressLine2</th>
      <th>RegAddress.PostTown</th>
      <th>RegAddress.County</th>
      <th>RegAddress.Country</th>
      <th>RegAddress.PostCode</th>
      <th>CompanyCategory</th>
      <th>CompanyStatus</th>
      <th>CountryOfOrigin</th>
      <th>DissolutionDate</th>
      <th>IncorporationDate</th>
      <th>Accounts.AccountRefDay</th>
      <th>Accounts.AccountRefMonth</th>
      <th>Accounts.NextDueDate</th>
      <th>Accounts.LastMadeUpDate</th>
      <th>Accounts.AccountCategory</th>
      <th>Returns.NextDueDate</th>
      <th>Returns.LastMadeUpDate</th>
      <th>Mortgages.NumMortCharges</th>
      <th>Mortgages.NumMortOutstanding</th>
      <th>Mortgages.NumMortPartSatisfied</th>
      <th>Mortgages.NumMortSatisfied</th>
      <th>SICCode.SicText_1</th>
      <th>SICCode.SicText_2</th>
      <th>SICCode.SicText_3</th>
      <th>SICCode.SicText_4</th>
      <th>LimitedPartnerships.NumGenPartners</th>
      <th>LimitedPartnerships.NumLimPartners</th>
      <th>URI</th>
      <th>PreviousName_1.CONDATE</th>
      <th>PreviousName_1.CompanyName</th>
      <th>PreviousName_2.CONDATE</th>
      <th>PreviousName_2.CompanyName</th>
      <th>PreviousName_3.CONDATE</th>
      <th>PreviousName_3.CompanyName</th>
      <th>PreviousName_4.CONDATE</th>
      <th>PreviousName_4.CompanyName</th>
      <th>PreviousName_5.CONDATE</th>
      <th>PreviousName_5.CompanyName</th>
      <th>PreviousName_6.CONDATE</th>
      <th>PreviousName_6.CompanyName</th>
      <th>PreviousName_7.CONDATE</th>
      <th>PreviousName_7.CompanyName</th>
      <th>PreviousName_8.CONDATE</th>
      <th>PreviousName_8.CompanyName</th>
      <th>PreviousName_9.CONDATE</th>
      <th>PreviousName_9.CompanyName</th>
      <th>PreviousName_10.CONDATE</th>
      <th>PreviousName_10.CompanyName</th>
      <th>ConfStmtNextDueDate</th>
      <th>ConfStmtLastMadeUpDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>! LTD</td>
      <td>08209948</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>METROHOUSE 57 PEPPER ROAD</td>
      <td>HUNSLET</td>
      <td>LEEDS</td>
      <td>YORKSHIRE</td>
      <td>NaN</td>
      <td>LS10 2RU</td>
      <td>Private Limited Company</td>
      <td>Active</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>11/09/2012</td>
      <td>30.0</td>
      <td>9.0</td>
      <td>30/06/2018</td>
      <td>30/09/2016</td>
      <td>DORMANT</td>
      <td>09/10/2016</td>
      <td>11/09/2015</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99999 - Dormant Company</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>http://business.data.gov.uk/id/company/08209948</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25/09/2019</td>
      <td>11/09/2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>!NNOV8 LIMITED</td>
      <td>11006939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C/O FRANK HIRTH 1ST FLOOR</td>
      <td>236 GRAY'S INN ROAD</td>
      <td>LONDON</td>
      <td>NaN</td>
      <td>UNITED KINGDOM</td>
      <td>WC1X 8HB</td>
      <td>Private Limited Company</td>
      <td>Active</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>11/10/2017</td>
      <td>31.0</td>
      <td>3.0</td>
      <td>11/07/2019</td>
      <td>NaN</td>
      <td>NO ACCOUNTS FILED</td>
      <td>08/11/2018</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>62090 - Other information technology service a...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>http://business.data.gov.uk/id/company/11006939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24/10/2019</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>!NSPIRED LTD</td>
      <td>SC421617</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26 POLMUIR ROAD</td>
      <td>NaN</td>
      <td>ABERDEEN</td>
      <td>NaN</td>
      <td>UNITED KINGDOM</td>
      <td>AB11 7SY</td>
      <td>Private Limited Company</td>
      <td>Active</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>11/04/2012</td>
      <td>30.0</td>
      <td>3.0</td>
      <td>30/12/2018</td>
      <td>30/03/2017</td>
      <td>TOTAL EXEMPTION FULL</td>
      <td>09/05/2017</td>
      <td>11/04/2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70229 - Management consultancy activities othe...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>http://business.data.gov.uk/id/company/SC421617</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25/04/2020</td>
      <td>11/04/2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>!NVERTD DESIGNS LIMITED</td>
      <td>09152972</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55A HIGH STREET</td>
      <td>NaN</td>
      <td>SILSOE</td>
      <td>BEDFORDSHIRE</td>
      <td>NaN</td>
      <td>MK45 4EW</td>
      <td>Private Limited Company</td>
      <td>Active</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>30/07/2014</td>
      <td>31.0</td>
      <td>7.0</td>
      <td>30/04/2019</td>
      <td>31/07/2017</td>
      <td>NaN</td>
      <td>27/08/2016</td>
      <td>30/07/2015</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>58190 - Other publishing activities</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>http://business.data.gov.uk/id/company/09152972</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13/08/2020</td>
      <td>30/07/2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>!OBAC LIMITED</td>
      <td>FC031362</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1ST AND 2ND FLOORS ELIZABETH HOUSE</td>
      <td>LES RUETIES BRAYES</td>
      <td>ST PETER PORT</td>
      <td>GUERNSEY</td>
      <td>GUERNSEY</td>
      <td>GY1 1EW</td>
      <td>Other company type</td>
      <td>Active</td>
      <td>CHANNEL ISLANDS</td>
      <td>NaN</td>
      <td>30/11/2012</td>
      <td>31.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>31/12/2016</td>
      <td>GROUP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None Supplied</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>http://business.data.gov.uk/id/company/FC031362</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

*This is a huge table with lots of rows, it may take a while to load*

```python
df.columns
```




    Index(['CompanyName', ' CompanyNumber', 'RegAddress.CareOf',
           'RegAddress.POBox', 'RegAddress.AddressLine1',
           ' RegAddress.AddressLine2', 'RegAddress.PostTown', 'RegAddress.County',
           'RegAddress.Country', 'RegAddress.PostCode', 'CompanyCategory',
           'CompanyStatus', 'CountryOfOrigin', 'DissolutionDate',
           'IncorporationDate', 'Accounts.AccountRefDay',
           'Accounts.AccountRefMonth', 'Accounts.NextDueDate',
           'Accounts.LastMadeUpDate', 'Accounts.AccountCategory',
           'Returns.NextDueDate', 'Returns.LastMadeUpDate',
           'Mortgages.NumMortCharges', 'Mortgages.NumMortOutstanding',
           'Mortgages.NumMortPartSatisfied', 'Mortgages.NumMortSatisfied',
           'SICCode.SicText_1', 'SICCode.SicText_2', 'SICCode.SicText_3',
           'SICCode.SicText_4', 'LimitedPartnerships.NumGenPartners',
           'LimitedPartnerships.NumLimPartners', 'URI', 'PreviousName_1.CONDATE',
           ' PreviousName_1.CompanyName', ' PreviousName_2.CONDATE',
           ' PreviousName_2.CompanyName', 'PreviousName_3.CONDATE',
           ' PreviousName_3.CompanyName', 'PreviousName_4.CONDATE',
           ' PreviousName_4.CompanyName', 'PreviousName_5.CONDATE',
           ' PreviousName_5.CompanyName', 'PreviousName_6.CONDATE',
           ' PreviousName_6.CompanyName', 'PreviousName_7.CONDATE',
           ' PreviousName_7.CompanyName', 'PreviousName_8.CONDATE',
           ' PreviousName_8.CompanyName', 'PreviousName_9.CONDATE',
           ' PreviousName_9.CompanyName', 'PreviousName_10.CONDATE',
           ' PreviousName_10.CompanyName', 'ConfStmtNextDueDate',
           ' ConfStmtLastMadeUpDate'],
          dtype='object')




```python
df['RegAddress.PostTown'].value_counts().head(30)
```




    LONDON                 767269
    MANCHESTER              71854
    BIRMINGHAM              70114
    GLASGOW                 51945
    BRISTOL                 47288
    EDINBURGH               45811
    LEEDS                   40947
    LIVERPOOL               33040
    NOTTINGHAM              32368
    LEICESTER               31233
    SHEFFIELD               26803
    WARRINGTON              24281
    BRIGHTON                23064
    HARROW                  22026
    CARDIFF                 21907
    COVENTRY                21771
    READING                 21461
    MILTON KEYNES           20348
    SOUTHAMPTON             19225
    ILFORD                  18380
    NORWICH                 17227
    STOCKPORT               17116
    NORTHAMPTON             16263
    CROYDON                 16013
    BOLTON                  15950
    CAMBRIDGE               15889
    BELFAST                 15627
    NEWCASTLE UPON TYNE     15569
    DERBY                   15241
    POOLE                   14982
    Name: RegAddress.PostTown, dtype: int64



## Frequency of words
Since we have lots of companies, we will only use companies in Cambridge as an example.

First we find the 30 most common words in all company names. As we will be expecting them to be repeating a lot even in companies that is not the same, we cannot match company names using them. The way we do it is we will deduct the matching score of a pair if any keywords is present in the names.


```python
from collections import Counter
all_names = df['CompanyName'][df['RegAddress.PostTown']=='CAMBRIDGE'].unique()
names_freq = Counter()
for name in all_names:
    names_freq.update(str(name).split(" "))
key_words = [word for (word,_) in names_freq.most_common(30)]
print(key_words)
```

    ['LIMITED', 'LTD', 'CAMBRIDGE', 'SERVICES', 'MANAGEMENT', '&', 'COMPANY', 'THE', 'CONSULTING', 'LTD.', 'SOLUTIONS', 'AND', 'PROPERTY', 'UK', 'LLP', '(CAMBRIDGE)', 'CONSULTANCY', 'GROUP', 'HOLDINGS', 'CONSULTANTS', 'ASSOCIATES', 'COMPOSITES', 'ENGINEERING', 'DEVELOPMENTS', 'INTERNATIONAL', 'OF', 'DESIGN', 'TECHNOLOGY', 'PROPERTIES', '(UK)']
    


```python
len(all_names)
```




    15889



## Matching by Grouping
Then we group the names by their 1st character. As the list is too long, it will take forever to match them all at once (15889 x 15889 pairs to consider). The work around is to match them by groups, assuming if the names are not matched at the 1st character, it is unlikely that they are the same name. 


```python
all_main_name = pd.DataFrame(columns=['sort_gp','names','alias','score'])
all_names.sort()
all_main_name['names'] = all_names
all_main_name['sort_gp'] = all_main_name['names'].apply(lambda x: x[0])
```

## Fuzzy Matching
Here for each group, we use `fuzzywuzzy.token_sort_ratio` to matching the names. Different form the basic `fuzzywuzzy.ratio` which use Levenshtein Distance to calculate the differences, it allow the token (words) in a name to swap order and still give a 'perfect' match. (ref: https://github.com/seatgeek/fuzzywuzzy)


```python
from fuzzywuzzy import fuzz

all_sort_gp = all_main_name['sort_gp'].unique()

def no_key_word(name):
    """check if the name contain the keywords in travel company"""
    output = True
    for key in key_words:
        if key in name:
            output = False
    return output

for sortgp in all_sort_gp:
    this_gp = all_main_name.groupby(['sort_gp']).get_group(sortgp)
    gp_start = this_gp.index.min()
    gp_end = this_gp.index.max()
    for i in range(gp_start,gp_end+1):
    
        # if self has not got alias, asign to be alias of itself
        if pd.isna(all_main_name['alias'].iloc[i]):
            all_main_name['alias'].iloc[i] = all_main_name['names'].iloc[i]
            all_main_name['score'].iloc[i] = 100
        
        # if the following has not got alias and fuzzy match, asign to be alias of this one
        for j in range(i+1,gp_end+1):
            if pd.isna(all_main_name['alias'].iloc[j]):
                fuzz_socre = fuzz.token_sort_ratio(all_main_name['names'].iloc[i],all_main_name['names'].iloc[j])
                if not no_key_word(all_main_name['names'].iloc[j]):
                    fuzz_socre -= 10
                if (fuzz_socre > 85):
                    all_main_name['alias'].iloc[j] = all_main_name['alias'].iloc[i]
                    all_main_name['score'].iloc[j] = fuzz_socre
                    
        if i % (len(all_names)//10) == 0:
            print("progress: %.2f" % (100*i/len(all_names)) + "%")
                
all_main_name.to_csv('company_in_cambridge.csv')
```

    progress: 0.00%
    progress: 9.99%
    progress: 19.99%
    progress: 29.98%
    progress: 39.98%
    progress: 49.97%
    progress: 59.97%
    progress: 69.96%
    progress: 79.95%
    progress: 89.95%
    progress: 99.94%
    


```python
all_main_name[(all_main_name['names']!=all_main_name['alias']) & (all_main_name['alias'].notna())]
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sort_gp</th>
      <th>names</th>
      <th>alias</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>761</th>
      <td>A</td>
      <td>AMADEUS EII LP</td>
      <td>AMADEUS EI LP</td>
      <td>96</td>
    </tr>
    <tr>
      <th>762</th>
      <td>A</td>
      <td>AMADEUS EIII LP</td>
      <td>AMADEUS EI LP</td>
      <td>93</td>
    </tr>
    <tr>
      <th>763</th>
      <td>A</td>
      <td>AMADEUS HI LP</td>
      <td>AMADEUS EI LP</td>
      <td>92</td>
    </tr>
    <tr>
      <th>766</th>
      <td>A</td>
      <td>AMADEUS II 'A'</td>
      <td>AMADEUS I</td>
      <td>86</td>
    </tr>
    <tr>
      <th>767</th>
      <td>A</td>
      <td>AMADEUS II 'B'</td>
      <td>AMADEUS I</td>
      <td>86</td>
    </tr>
    <tr>
      <th>768</th>
      <td>A</td>
      <td>AMADEUS II 'C'</td>
      <td>AMADEUS I</td>
      <td>86</td>
    </tr>
    <tr>
      <th>769</th>
      <td>A</td>
      <td>AMADEUS III</td>
      <td>AMADEUS I</td>
      <td>90</td>
    </tr>
    <tr>
      <th>773</th>
      <td>A</td>
      <td>AMADEUS IV EARLY STAGE FUND B LP</td>
      <td>AMADEUS IV EARLY STAGE FUND A LP</td>
      <td>94</td>
    </tr>
    <tr>
      <th>776</th>
      <td>A</td>
      <td>AMADEUS JI LP</td>
      <td>AMADEUS EI LP</td>
      <td>92</td>
    </tr>
    <tr>
      <th>777</th>
      <td>A</td>
      <td>AMADEUS LI LP</td>
      <td>AMADEUS EI LP</td>
      <td>92</td>
    </tr>
    <tr>
      <th>783</th>
      <td>A</td>
      <td>AMADEUS TI LP</td>
      <td>AMADEUS PI LP</td>
      <td>92</td>
    </tr>
    <tr>
      <th>937</th>
      <td>A</td>
      <td>ANP ENGINEERING LIMITED</td>
      <td>AMP ENGINEERING LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>A</td>
      <td>AVITO LIMITED</td>
      <td>AVIATO LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>1607</th>
      <td>B</td>
      <td>BCK INVESTMENTS LIMITED</td>
      <td>BCK INVESTMENTS (2) LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>1849</th>
      <td>B</td>
      <td>BIRDY FILMS LTD</td>
      <td>BIRD FILMS LTD</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>B</td>
      <td>BLUESTONE CAPITAL MANAGEMENT LIMITED</td>
      <td>BLUESTONE CAPITAL MANAGEMENT II LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>B</td>
      <td>BLUESTONE MORTGAGE FINANCE NO. 4 LIMITED</td>
      <td>BLUESTONE MORTGAGE FINANCE NO. 3 LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>B</td>
      <td>BLUESTONE MORTGAGE FINANCE NO.2 LIMITED</td>
      <td>BLUESTONE MORTGAGE FINANCE NO. 3 LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2698</th>
      <td>C</td>
      <td>CAMBRIDGE DESIGN PARTNERSHIP LIMITED</td>
      <td>CAMBRIDGE DESIGN PARTNERSHIP (UK) LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>C</td>
      <td>CAMBRIDGE H C 102 LIMITED</td>
      <td>CAMBRIDGE H C 100 LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2840</th>
      <td>C</td>
      <td>CAMBRIDGE H C 129 LIMITED</td>
      <td>CAMBRIDGE H C 100 LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2911</th>
      <td>C</td>
      <td>CAMBRIDGE INTERNATIONAL EDUCATION CENTRE LIMITED</td>
      <td>CAMBRIDGE EDUCATION INTERNATIONAL CENTER LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>C</td>
      <td>CAMBRIDGE RESEARCH AND ENGINEERING LIMITED</td>
      <td>CAMBRIDGE ENGINEERING AND RESEARCH LIMITED</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3184</th>
      <td>C</td>
      <td>CAMBRIDGE SEMINARS INTERNATIONAL COLLEGE LIMITED</td>
      <td>CAMBRIDGE INTERNATIONAL SEMINARS COLLEGE LTD</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3566</th>
      <td>C</td>
      <td>CASA MIA DEVELOPMENTS (NO 3) LIMITED</td>
      <td>CASA MIA DEVELOPMENTS (NO 1) LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4159</th>
      <td>C</td>
      <td>COLLEGE COURT (CAMBRIDGE B) LIMITED</td>
      <td>COLLEGE COURT (CAMBRIDGE A) LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4944</th>
      <td>D</td>
      <td>DGP ENGINEERING SERVICES LTD</td>
      <td>DG ENGINEERING SERVICES LTD</td>
      <td>88</td>
    </tr>
    <tr>
      <th>5182</th>
      <td>D</td>
      <td>DS ELECTRICAL (CAMBRIDGE) LIMITED</td>
      <td>DBS ELECTRICAL (CAMBRIDGE) LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>6220</th>
      <td>F</td>
      <td>FRAZER STANNARD PROPERTY LTD</td>
      <td>FRASER STANNARD PROPERTY LTD</td>
      <td>86</td>
    </tr>
    <tr>
      <th>6407</th>
      <td>G</td>
      <td>GATEWAY (218) MANAGEMENT COMPANY LIMITED</td>
      <td>GATEWAY (216) MANAGEMENT COMPANY LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>6517</th>
      <td>G</td>
      <td>GFB AERO LTD</td>
      <td>GB AERO LTD</td>
      <td>86</td>
    </tr>
    <tr>
      <th>6951</th>
      <td>H</td>
      <td>HAMMER AND TONG PRODUCTIONS LIMITED</td>
      <td>HAMMER AND THONGS PRODUCTIONS LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>7354</th>
      <td>H</td>
      <td>HPG DEVELOPMENTS LIMITED</td>
      <td>HP3 DEVELOPMENTS LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>7811</th>
      <td>I</td>
      <td>IQ CAPITAL FUND II LP</td>
      <td>IQ CAPITAL FUND I LP</td>
      <td>98</td>
    </tr>
    <tr>
      <th>7812</th>
      <td>I</td>
      <td>IQ CAPITAL FUND III C LP</td>
      <td>IQ CAPITAL FUND I LP</td>
      <td>91</td>
    </tr>
    <tr>
      <th>7813</th>
      <td>I</td>
      <td>IQ CAPITAL FUND III LP</td>
      <td>IQ CAPITAL FUND I LP</td>
      <td>95</td>
    </tr>
    <tr>
      <th>7817</th>
      <td>I</td>
      <td>IQ CAPITAL PARTNERS GP III LLP</td>
      <td>IQ CAPITAL PARTNERS GP III C LLP</td>
      <td>87</td>
    </tr>
    <tr>
      <th>8420</th>
      <td>K</td>
      <td>KEW BRIDGE WEST 2 RESIDENTS MANAGEMENT COMPANY...</td>
      <td>KEW BRIDGE WEST 1 RESIDENTS MANAGEMENT COMPANY...</td>
      <td>88</td>
    </tr>
    <tr>
      <th>8421</th>
      <td>K</td>
      <td>KEW BRIDGE WEST 3 RESIDENTS MANAGEMENT COMPANY...</td>
      <td>KEW BRIDGE WEST 1 RESIDENTS MANAGEMENT COMPANY...</td>
      <td>88</td>
    </tr>
    <tr>
      <th>9772</th>
      <td>M</td>
      <td>MILLERS PARTNERS LIMITED</td>
      <td>MILLER PARTNERS LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>11225</th>
      <td>P</td>
      <td>PIGEON HUNSTANTON 2 LIMITED</td>
      <td>PIGEON HUNSTANTON 1 LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>11711</th>
      <td>Q</td>
      <td>QQ2 RESIDENTS PROPERTY MANAGEMENT LIMITED</td>
      <td>QQ1 RESIDENTS PROPERTY MANAGEMENT LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>12129</th>
      <td>R</td>
      <td>RIDGEMOND PARK (BLOCK 4) MANAGEMENT COMPANY LI...</td>
      <td>RIDGEMOND PARK (BLOCK 3) MANAGEMENT COMPANY LI...</td>
      <td>88</td>
    </tr>
    <tr>
      <th>12156</th>
      <td>R</td>
      <td>RINDL CONSULTING LIMITED</td>
      <td>RIL CONSULTING LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>12523</th>
      <td>S</td>
      <td>SANDFORD AVIATION LTD</td>
      <td>SANDFORD AVIAITION LTD</td>
      <td>88</td>
    </tr>
    <tr>
      <th>12807</th>
      <td>S</td>
      <td>SHAWS OF CAMBRIDGE (EH) LIMITED</td>
      <td>SHAWS OF CAMBRIDGE (EB) LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>12902</th>
      <td>S</td>
      <td>SILKWORKS RESIDENTS MANAGEMENT COMPANY 3 LIMITED</td>
      <td>SILKWORKS RESIDENTS MANAGEMENT COMPANY 2 LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>13499</th>
      <td>S</td>
      <td>STOW HEALTHCARE (FP) LIMITED</td>
      <td>STOW HEALTHCARE (BP) LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>13865</th>
      <td>T</td>
      <td>TCOM LIMITED</td>
      <td>TCO LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>14116</th>
      <td>T</td>
      <td>THE CUWBC FOUNDATION</td>
      <td>THE CUBC FOUNDATION</td>
      <td>87</td>
    </tr>
    <tr>
      <th>14149</th>
      <td>T</td>
      <td>THE EXETER TRUSTEE COMPANY NO.2 LIMITED</td>
      <td>THE EXETER TRUSTEE COMPANY NO. 1 LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>14343</th>
      <td>T</td>
      <td>THE PROCUREMENT PARTNERSHIP LIMITED</td>
      <td>THE PROCUREMENT PARTNERS LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>14702</th>
      <td>T</td>
      <td>TRANQUILLITY PARKS LIMITED</td>
      <td>TRANQUILITY PARKS LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>14816</th>
      <td>T</td>
      <td>TSANG CONSULTING LIMITED</td>
      <td>TAG CONSULTING LIMITED</td>
      <td>86</td>
    </tr>
    <tr>
      <th>15340</th>
      <td>W</td>
      <td>WATER LANE 3 NOMINEES LIMITED</td>
      <td>WATER LANE 2 NOMINEES LIMITED</td>
      <td>87</td>
    </tr>
    <tr>
      <th>15349</th>
      <td>W</td>
      <td>WATERBEACH NO.1 MANAGEMENT COMPANY LIMITED</td>
      <td>WATERBEACH NO. 2 MANAGEMENT COMPANY LIMITED</td>
      <td>88</td>
    </tr>
    <tr>
      <th>15667</th>
      <td>W</td>
      <td>WRENBRIDGE (CHEDDARS LANE) LIMITED</td>
      <td>WRENBRIDGE (CHEDDARS LANE 2) LIMITED</td>
      <td>87</td>
    </tr>
  </tbody>
</table>

The result is saved in a csv file locally for future inspection and further experimentation. Inspecting the result, the matches consisted of 3 groups:

1. they are usually differ in spelling by 1 character: missing an 'L' or 'I' or 'S'
2. highly similar names: 'No.3' instead of 'No.2' or 'EB' instread of 'EH'
3. fairly similar names: 'HAMMER AND THONGS PRODUCTIONS LIMITED' and 'HAMMER AND TONG PRODUCTIONS LIMITED'

For type 1 and 2 matches it could be the same company, the diffeernce in names could be an intentional alteration or simply a typo. But it is not likely the same company for type 3 matched, it seems more like a coincidnce. 

To further confirm, manual work need to be done but this program saves a lot of manual work hours.

```python
all_main_name[(all_main_name['names']!=all_main_name['alias']) & (all_main_name['alias'].notna())].shape[0]
```




    57




```python
len(all_main_name['alias'].unique())
```




    15832



By applying the fuzzy matching, 57 names are caught similar to another name, which is less then 1% of the total. By using this program names that need checking drastically reduce form 15889 total to only 57.
