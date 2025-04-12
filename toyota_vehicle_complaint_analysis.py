#!/usr/bin/env python
# coding: utf-8

# <h1>Safety Complaints Analysis</h1>

# In[3]:


#importing required libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from collections import Counter
import geopandas as gpd
from langchain.llms import Ollama
import json
import requests


# **Note**: The dataset contains safety complaints from vehicle users in the U.S and it's downloaded from NHTSA Website (Link: https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip)

# ## NHTSA Dataset 
# Complaint information entered into NHTSA’s Office of Defects Investigation vehicle owner's complaint database is used with other data sources to identify safety issues that warrant investigation and to determine if a safety-related defect trend exists. Complaint information is also analyzed to monitor existing recalls for proper scope and adequacy.
# 
# 
# <h3>Column Descriptions</h3>
# 
# 1. **CMPLID** - NHTSA's internal unique sequence number (Updatable field).
# 2. **ODINO** - NHTSA's internal reference number (may repeat for multiple components).
# 3. **MFR_NAME** - Manufacturer's name.
# 4. **MAKETXT** - Vehicle/Equipment make.
# 5. **MODELTXT** - Vehicle/Equipment model.
# 6. **YEARTXT** - Model year, '9999' if unknown or N/A.
# 7. **CRASH** - Was vehicle involved in a crash? ('Y' or 'N')
# 8. **FAILDATE** - Date of incident (YYYYMMDD).
# 9. **FIRE** - Was vehicle involved in a fire? ('Y' or 'N')
# 10. **INJURED** - Number of persons injured.
# 11. **DEATHS** - Number of fatalities.
# 12. **COMPDESC** - Specific component's description.
# 13. **CITY** - Consumer's city.
# 14. **STATE** - Consumer's state code.
# 15. **VIN** - Vehicle's VIN number.
# 16. **DATEA** - Date added to file (YYYYMMDD).
# 17. **LDATE** - Date complaint received by NHTSA (YYYYMMDD).
# 18. **MILES** - Vehicle mileage at failure.
# 19. **OCCURENCES** - Number of occurrences.
# 20. **CDESCR** - Description of the complaint.
# 21. **CMPL_TYPE** - Source of complaint code (CAG, CON, DP, EVOQ, etc.).
# 22. **POLICE_RPT_YN** - Was incident reported to police? ('Y' or 'N')
# 23. **PURCH_DT** - Date purchased (YYYYMMDD).
# 24. **ORIG_OWNER_YN** - Was original owner? ('Y' or 'N')
# 25. **ANTI_BRAKES_YN** - Anti-lock brakes? ('Y' or 'N')
# 26. **CRUISE_CONT_YN** - Cruise control? ('Y' or 'N')
# 27. **NUM_CYLS** - Number of cylinders.
# 28. **DRIVE_TRAIN** - Drive train type (AWD, 4WD, FWD, RWD).
# 29. **FUEL_SYS** - Fuel system code (FI, TB).
# 30. **FUEL_TYPE** - Fuel type code (BF, CN, DS, GS, HE).
# 31. **TRANS_TYPE** - Vehicle transmission type (AUTO, MAN).
# 32. **VEH_SPEED** - Vehicle speed.
# 33. **DOT** - Department of Transportation tire identifier.
# 34. **TIRE_SIZE** - Tire size.
# 35. **LOC_OF_TIRE** - Location of tire code (FSW, DSR, FTR, PSR, SPR).
# 36. **TIRE_FAIL_TYPE** - Type of tire failure code (BST, BLW, TTL, OFR, TSW, TTR, TSP).
# 37. **ORIG_EQUIP_YN** - Was part original equipment? ('Y' or 'N')
# 38. **MANUF_DT** - Date of manufacture (YYYYMMDD).
# 39. **SEAT_TYPE** - Type of child seat code (B, C, I, IN, TD).
# 40. **RESTRAINT_TYPE** - Installation system code (A = Safety belt, B = LATCH system).
# 41. **DEALER_NAME** - Dealer's name.
# 42. **DEALER_TEL** - Dealer's telephone number.
# 43. **DEALER_CITY** - Dealer's city.
# 44. **DEALER_STATE** - Dealer's state code.
# 45. **DEALER_ZIP** - Dealer's ZIP code.
# 46. **PROD_TYPE** - Product type code (V = Vehicle, T = Tires, E = Equipment, C = Child Restraint).
# 47. **REPAIRED_YN** - Was defective tire repaired? ('Y' or 'N')
# 48. **MEDICAL_ATTN** - Was medical attention required? ('Y' or 'N')
# 49. **VEHICLES_TOWED_YN** - Was vehicle towed? ('Y' or 'N')
# 

# In[6]:


#adding column names
column_names = ['CMPLID', 'ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT',
               'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED',
               'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN',
               'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR',
               'CMPL_TYPE', 'POLICE_RPT_YN','PURCH_DT', 'ORIG_OWNER_YN',
               'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS',
               'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE', 'TRANS_TYPE',
               'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE',
               'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE',
               'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE',
               'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN',
               'VEHICLES_TOWED_YN'] 


# In[7]:


# loading dataset
df_main = pd.read_csv('FLAT_CMPL.txt', delimiter='\t', names=column_names, header=None, on_bad_lines='skip')
df_main = df_main[:]
print(df_main.shape)
print(df_main['CMPLID'].nunique())


# **Note**: I have identified that when multiple systems (COMPDESC) are invloved, the complaints were duplicated with same ODINO. So in the below cell, I have grouped the records with respect to ODINO to eliminate duplicate records

# In[9]:


#Handling duplicate records
df_dedup1 = df_main[['ODINO', 'COMPDESC']]
df_dedup = df_dedup1.groupby('ODINO').agg(lambda x: ','.join(x.astype(str).dropna()) if x.notna().any() else None).reset_index()
# df.drop(columns=['CMPLID'], inplace=True)


def remove_duplicates(dataset, column_to_process):
        def remove_duplicate_tags(tag_string):
            if pd.isna(tag_string) or not isinstance(tag_string, str):
                return None

            tag_list = [tag.strip() for tag in tag_string.split(', ')]
            seen = set()
            unique_tags = [tag for tag in tag_list if not (tag in seen or seen.add(tag))]
            return ','.join(unique_tags)

        dataset[column_to_process] = dataset[column_to_process].astype(str).apply(remove_duplicate_tags)

for col in df_dedup.columns:
    remove_duplicates(df_dedup, col)

df_dedup.head(1)


# In[10]:


df1 = df_main.copy()
df1.drop(columns=['CMPLID','COMPDESC'], inplace=True)
df1['ODINO'] = df1['ODINO'].astype(str)
df1.drop_duplicates(subset='ODINO', inplace=True)
df = pd.merge(df1, df_dedup, on='ODINO', how='left')
df.head(1)


# In[11]:


df['FAILDATE'] = pd.to_datetime(df['FAILDATE'], format='%Y%m%d', errors='coerce')
df['DATEA'] = pd.to_datetime(df['DATEA'], format='%Y%m%d', errors='coerce')
df['LDATE'] = pd.to_datetime(df['LDATE'], format='%Y%m%d', errors='coerce')
df['MANUF_DT'] = pd.to_datetime(df['MANUF_DT'], format='%Y%m%d', errors='coerce')


# ## Analyzing Toyota Safety Complaints
# (Year: 2020-2024)

# In[13]:


#filtering Toyota complaints
dataframe = df[df['MFR_NAME']=='Toyota Motor Corporation']
dataframe = dataframe[dataframe['LDATE'].dt.year>=2020]
dataframe = dataframe[dataframe['LDATE'].dt.year!=9999]
dataframe = dataframe[dataframe['LDATE'].dt.year<2025]
dataframe = dataframe[:]
dataframe.head(1)


# In[14]:


import plotly.graph_objects as go

# Calculate non-null percentage
non_null_percentage = (dataframe.notnull().sum() / len(dataframe)) * 100

# Create the bar chart
fig = px.bar(
    x=non_null_percentage.values,
    y=non_null_percentage.index,
    orientation='h',
    labels={'x': 'Non-Null Percentage (%)', 'y': 'Column Name'},
    title='Population (% of Non-Null Values) for Each Column',
    color_discrete_sequence=['#FF5C5C']
)

fig.update_traces(marker_line_color='black', marker_line_width=1)

# Update layout for better aesthetics
fig.update_layout(
    title='Population (% of Non-Null Values) for Each Column',
    width=900,
    height=900,
    xaxis_title='Non-Null Percentage (%)',
    yaxis_title='Column Names',
    xaxis=dict(range=[0, 105], gridcolor='lightgray'),  # Ensure space for labels above bars
    plot_bgcolor='#58595B'
)

# Show the plot
fig.show()


# In[15]:


print("\033[1mToyota 🚗\033[0m")
print(f' - {dataframe.shape[0]} safety complaints 🦺 have been registered between 2020 and 2024')
print('\n')
# Count the frequency of each unique value

frequency = pd.to_datetime(dataframe['LDATE']).dt.year.value_counts().reset_index()
frequency.columns = ['Year', 'Number of Complaints']

# Create a bar plot using Plotly
fig = px.bar(frequency, x='Year', y='Number of Complaints', title='Annual Number of Complaints',
            color_discrete_sequence=['#FF5C5C'])

fig.update_traces(marker_line_color='black', marker_line_width=1)

fig.update_layout(
    width=500,
    height=300,
    plot_bgcolor='#58595B',
    paper_bgcolor='white',
    font=dict(color='black'),
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),

)

# Show the plot
fig.show()


# In[16]:


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming 'dataframe' is already defined with columns 'MAKETXT' and 'MODELTXT'
temp = dataframe.copy()

# Calculate frequency for makes
make_freq = temp['MAKETXT'].value_counts().reset_index()
make_freq.columns = ['Make', 'Count']
make_freq = make_freq.sort_values(by='Count', ascending=False)  # Sort in decreasing order

# Calculate frequency for models along with corresponding makes
model_freq = temp.groupby(['MODELTXT', 'MAKETXT']).size().reset_index(name='Count')
model_freq = model_freq.sort_values(by='Count', ascending=False)  # Sort in decreasing order

# Select top 15 models
top_10_model_freq = model_freq.head(15)

# Create subplots
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("Vehicle Make Complaints", "Top 15 Vehicle Model Complaints"))

# Add bar plot for Vehicle Make
fig.add_trace(
    go.Bar(x=make_freq['Make'], y=make_freq['Count'], name='Vehicle Make', 
           marker_color='#FF5C5C', marker_line_color='black', marker_line_width=1),
    row=1, col=1
)

# Add bar plot for Top 15 Vehicle Models with Make information in the hover
fig.add_trace(
    go.Bar(
        x=top_10_model_freq['MODELTXT'], 
        y=top_10_model_freq['Count'], 
        name='Vehicle Model', 
        marker_color='#FF5C5C', 
        marker_line_color='black', 
        marker_line_width=1,
        hovertemplate='<b>Model:</b> %{x}<br><b>Make:</b> %{customdata[0]}<br><b>Complaints:</b> %{y}',
        customdata=top_10_model_freq[['MAKETXT']].values  # Passing the make as custom data for hover
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1000,
    height=500,
    plot_bgcolor='#58595B',
    paper_bgcolor='white',
    font=dict(color='black'),
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),

)

# Add x-axis and y-axis titles
fig.update_xaxes(title_text="Vehicle Make", tickangle=-45, row=1, col=1)
fig.update_yaxes(title_text="Number of Complaints", row=1, col=1)

fig.update_xaxes(title_text="Vehicle Model (Top 15)", tickangle=-45, row=1, col=2)
fig.update_yaxes(title_text="Number of Complaints", row=1, col=2)

fig.show()


# 
# **Note**: The above charts only show the number of complaints registered against each vehicle make and model. They don't convey any information about the most unsafe or safe vehicles, as the values are **not normalized**

# In[18]:


red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['#ffcccc', '#ff0000'])

# Clean and prepare text
cleaned_text_series = (
    dataframe['MODELTXT']
    .dropna()
    .str.upper()
    .str.strip()
    .str.replace(r'\s+|[^A-Z0-9]', '-', regex=True)  # Apply regex here
      # Remove duplicate entries
)

word_freq = Counter(cleaned_text_series)



# Generate the word cloud
plt.figure(figsize=(15, 15))
wordcloud = WordCloud(
    background_color="#58595B",
    max_words=100,
    max_font_size=150,
    width=1000,
    height=400,
    colormap=red_cmap,
    contour_color='black',  # Add black border
    contour_width=5 
).generate_from_frequencies(word_freq)

plt.title(
    "Word Cloud of Model",
    fontsize=15,
    color='black', 
    pad=10,  
    loc='left' 
)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[19]:


import plotly.graph_objects as go
import pandas as pd

# Selecting relevant columns
temp_df = dataframe[['CRASH', 'FIRE', 'INJURED', 'DEATHS', 'MEDICAL_ATTN', 'POLICE_RPT_YN']]
temp_df['CRASH'] = temp_df['CRASH'].replace('Y', 1)
temp_df['FIRE'] = temp_df['FIRE'].replace('Y', 1)
temp_df['MEDICAL_ATTN'] = temp_df['MEDICAL_ATTN'].replace('Y', 1)
temp_df['POLICE_RPT_YN'] = temp_df['POLICE_RPT_YN'].replace('Y', 1)

# Count of 'Y' values in each column
counts = temp_df.apply(lambda col: (col == 1).sum()).reset_index()
counts.columns = ['Category', 'Value']

# Create the plotly figure
fig = go.Figure()

# Add a card-like shape for each category
for i, row in counts.iterrows():
    fig.add_trace(go.Indicator(
        mode="number",
        value=row['Value'],
        number={'valueformat': ",", 'font': {'size': 36, 'color': '#FF5C5C'}},
        domain={'x': [i/len(counts), (i+1)/len(counts)], 'y': [0.2, 1]}
    ))
    
    # Adding category label using annotations
    fig.add_annotation(
        x=(i + 0.5) / len(counts),
        y=0.1,
        text=row['Category'],
        showarrow=False,
        font=dict(size=14, color='black'),
        xanchor='center'
    )

# Layout adjustments
fig.update_layout(title_text='Complaints reporting Crash, Fire, Injury, Death, Medical Attention incidents and Police Reported',
    grid={'rows': 1, 'columns': len(counts), 'pattern': "independent"},
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor='#f7f7f7'
)

fig.show()


# In[20]:


import plotly.express as px
import geopandas as gpd
import pandas as pd

# Reading the shapefile
cities = gpd.read_file(r'ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp')

# Assuming df_temp is already defined with relevant data
df_temp = dataframe.copy()
df_temp['CITY'] = df_temp['CITY'].str.title()

# Merging the data
df_1 = cities.merge(df_temp, how="left", left_on='postal', right_on='STATE')
# df_1 = df_1[df_1['ODINO'].notnull()]

# Counting the number of values in CMPLID column
df_1['Number of Complaints'] = df_1.groupby('name')['ODINO'].transform('count')

# Handling missing values
df_1['Number of Complaints'].fillna(0, inplace=True)
df = df.reset_index(drop=True)
# Ensure geometries are in GeoJSON format
df_1_json = df_1.__geo_interface__

# Plotting with Plotly
fig = px.choropleth(
    df_1,
    geojson=df_1_json,
    locations=df_1.index,
    color='Number of Complaints',
    color_continuous_scale=[[0, '#ffcccc'], [1, '#FF4C4C']],  # Custom red gradient
    hover_name='name',
    hover_data=['Number of Complaints']  # Only include this column
)

# Update layout
fig.update_geos(
    fitbounds="locations",
    visible=False,
    bgcolor='#58595B'
)

fig.update_layout(
    width=1000,
    height=500,
    margin=dict(l=0, r=0, t=60, b=0),  # Reduced padding
    title_text='Complaints by State',
    title_font_size=16,
    paper_bgcolor='white',  # Set figure background color
    plot_bgcolor='#58595B',   # Set plot background color
    coloraxis_colorbar=dict(
        title='Complaints',
        tickformat=',',  # To avoid scientific notation
    ),
    dragmode=False,  # Disable dragging
)

# Disable zoom interaction
fig.update_layout(
    geo=dict(
        projection_scale=1,  # Fixed scale
        center=dict(lat=0, lon=0),  # Keep centered
        lataxis=dict(range=[-90, 90]),
        lonaxis=dict(range=[-180, 180])
    )
)

fig.show()


# ## Anlayzing complaints that reported deaths

# In[22]:


dataframe_deaths = dataframe[dataframe['DEATHS']==1]
dataframe_deaths.shape


# In[23]:


frequency = dataframe_deaths['MODELTXT'].value_counts().reset_index()
frequency.columns = ['Model', 'Number of Complaints']

# Create a bar plot using Plotly
fig = px.bar(frequency, x='Model', y='Number of Complaints', title='Vehicle Models Involved In Death of Driver/Passengers',
            color_discrete_sequence=['#FF5C5C'])

fig.update_traces(marker_line_color='black', marker_line_width=1)

fig.update_layout(
    width=600,
    height=500,
    plot_bgcolor='#58595B',
    paper_bgcolor='white',
    font=dict(color='black'),
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(tickangle=-45)

)

# Show the plot
fig.show()


# ## Identifying the failed system that lead to the death of driver/passengers using Large Language Model

# In[25]:


Prompt_1 = '''As a data scientist with a background in mechanical engineering at Toyota Motor Corporation, your job is to extract the **precise root cause** from the issue description below.

The description outlines the reason behind a **fatal accident** involving a Toyota vehicle.

Instructions:
- Extract only the **root cause** of the accident.
- **Output must be a short phrase**, not a sentence.
- Do **not** include any model names, years, or any irrelevant details.
- Do **not** include "Toyota" or any manufacturer name.
- Be precise: **Component name + issue**, like:
  - "Brake Failure"
  - "Steering Lockup"
  - "Air Bag Malfunctioning"
  - "Loss of Control"
  - "Unidentified Mechanical Issue"
- Your output will be used as a categorical field for identifying systemic problems.

Be accurate — this information is critical to vehicle design safety.

Here is the description:
'''


ollama = Ollama(base_url="http://localhost:11434", model="mistral")
for index, row in dataframe_deaths.iterrows():
    input1 = Prompt_1 + row['CDESCR'].strip()
    dataframe_deaths.at[index, 'output_death'] = ollama.invoke(input1)
    
dataframe_deaths['output_death'] = dataframe_deaths['output_death'].str.strip()


# In[26]:


from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Words to visualize
words = dataframe_deaths['output_death'].unique()

# Load model and encode
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(words)

# Cluster the embeddings
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
reduced = tsne.fit_transform(embeddings)

# === Styling ===
plt.style.use('dark_background')
mpl.rcParams.update({
    'axes.facecolor': '#58595B',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'font.size': 16,           # General font size
    'xtick.labelsize': 14,     # X-axis tick label size
    'ytick.labelsize': 14,     # Y-axis tick label size
    'legend.fontsize': 14,     # Legend text size
})

plt.figure(figsize=(24, 14))  # Increased figure size
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
scatter_handles = []

for i in range(n_clusters):
    cluster_points = reduced[labels == i]
    cluster_words = np.array(words)[labels == i]

    # Scatter points
    scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          s=160, edgecolors='k', linewidth=1.2,
                          color=colors[i], label=f'Cluster {i+1}')
    scatter_handles.append(scatter)

    # Convex hull
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                     color=colors[i], linestyle='--', linewidth=1.5)

    # Word annotations
    for j, word in enumerate(cluster_words):
        x, y = cluster_points[j]
        plt.text(x + 0.5, y + 0.5, word, fontsize=14,
                 bbox=dict(facecolor='#FF5C5C', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.4'))

# Create the legend
legend = plt.legend(handles=scatter_handles, title='Clusters', frameon=True)

# Make all cluster labels red
for text in legend.get_texts():
    text.set_color('red')

# Style the legend background and border
legend.get_frame().set_facecolor('#f5f5f5')   # Light grey background
legend.get_frame().set_edgecolor('black')     # Border color

# Style the legend title
legend.get_title().set_color('darkred')       # Title text color
legend.get_title().set_fontweight('bold')     # Optional: make title bold
legend.get_title().set_fontsize(16)

# Final plot polish
plt.title("2D Semantic Word Map with Clusters", fontsize=20, color='black')
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=-45)
plt.tight_layout()
plt.show()


# ## 🔍 Key Insights from the Clustered Semantic Map
# 
# From the above 2D semantic clustering of system-related words, we can identify **three major categories of system failures** that contributed to the accident:
# 
# 1. 🛡️ **Airbag Malfunctioning** - Keywords in this cluster suggest issues with the airbag system's deployment or failure during critical moments.
# 2. 🛞 **Steering System Failure** - This group highlights problems related to steering control, potentially leading to loss of vehicle direction.
# 3. 🔋 **Battery and Electrical System Failures** - Includes terms associated with battery issues, wiring faults, or broader electrical malfunctions.
# 
# These clusters provide a valuable high-level view of the primary mechanical domains involved in the reported incidents.
# 

# In[28]:


## Anlayzing complaints that reported fire accident


# In[29]:


dataframe_fire = dataframe[dataframe['FIRE']=='Y']
dataframe_fire.shape


# In[30]:


frequency = dataframe_fire['MODELTXT'].value_counts().reset_index()
frequency.columns = ['Model', 'Number of Complaints']

# Create a bar plot using Plotly
fig = px.bar(frequency, x='Model', y='Number of Complaints', title='Vehicle Models Involved In Fire Incidents',
            color_discrete_sequence=['#FF5C5C'])

fig.update_traces(marker_line_color='black', marker_line_width=1)

fig.update_layout(
    width=800,
    height=500,
    plot_bgcolor='#58595B',
    paper_bgcolor='white',
    font=dict(color='black'),
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(tickangle=-45)

)

# Show the plot
fig.show()


# ## Identifying the failed system that lead to fire in the vehicle using Large Language Model

# In[32]:


Prompt_2 = '''As a data scientist with a background in mechanical engineering at Toyota Motor Corporation, your job is to extract the **precise root cause** from the issue description below.

The description outlines the reason behind a **fatal accident** involving a Toyota vehicle.

Instructions:
- Extract only the **root cause** of the accident.
- **Output must be a short phrase**, not a sentence.
- Do **not** include any model names, years, or any irrelevant details.
- Do **not** include "Toyota" or any manufacturer name.
- Be precise: **Component name + issue**, like:
  - "Brake Failure"
  - "Steering Lockup"
  - "Air Bag Malfunctioning"
  - "Loss of Control"
  - "Unidentified Mechanical Issue"
- Your output will be used as a categorical field for identifying systemic problems.

Be accurate — this information is critical to vehicle design safety.

Here is the description:
'''


ollama = Ollama(base_url="http://localhost:11434", model="mistral")
for index, row in dataframe_fire.iterrows():
    input1 = Prompt_2 + row['CDESCR'].strip()
    dataframe_fire.at[index, 'output_fire'] = ollama.invoke(input1)
    
dataframe_fire['output_fire'] = dataframe_fire['output_fire'].str.strip()


# In[33]:


red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['#ffcccc', '#ff0000'])

# Clean and prepare text
cleaned_text_series = (
    dataframe_fire['output_fire']
    .dropna()
    .str.upper()
    .str.strip()
    .str.replace(r'\s+|[^A-Z0-9]', ' ', regex=True)  # Apply regex here
      # Remove duplicate entries
)

word_freq = Counter(cleaned_text_series)



# Generate the word cloud
plt.figure(figsize=(15, 15))
wordcloud = WordCloud(
    background_color="#58595B",
    max_words=100,
    max_font_size=150,
    width=1000,
    height=400,
    colormap=red_cmap,
    contour_color='black',  # Add black border
    contour_width=5 
).generate_from_frequencies(word_freq)

plt.title(
    "Word Cloud",
    fontsize=15,
    color='black', 
    pad=10,  
    loc='left' 
)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## 🔍 Key Insights from the Word Cloud
# 
# From the above word cloud of system-related words, we can identify **three major categories of system failures** that contributed to the accident:
# 
# 1. ⚡ **Electrical System Failure** – This includes short circuits, wiring issues, faulty relays, and power distribution failures that disrupted system operations.
# 2. 🌡️ **Component Overheating** – Refers to thermal stress or inadequate cooling of parts such as motors, resistors, or connectors, often leading to performance degradation or permanent damage.
# 3. 🔥 **Battery Malfunction & Explosion** – Encompasses incidents of battery swelling, leakage, thermal runaway, or explosions due to overcharging, poor insulation, or manufacturing defects.
# 
# 

# Thank you for viewing the notebook!
# 
# Feel free to reach out with any questions or feedback.
