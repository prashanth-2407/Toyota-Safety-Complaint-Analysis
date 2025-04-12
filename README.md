# Toyota-Safety-Complaint-Analysis

In this project, I have analysed the safety complaints registered against Toyota's vehicle which is available in [NHTSA](https://www.nhtsa.gov/).

<h1>Some interesting insights drawn from the analysis</h1>

a) Number of safety complaints registered against toyota's vehicles from 2020 to 2024
   1. Year 2020: 4485 complaints
   2. Year 2021: 3223 complaints
   3. Year 2022: 2735 complaints
   4. Year 2023: 3079 complaints
   5. Year 2024: 4092 complaints
The average Year-over-year increase in complaints from 2022 onwards is **approximately 22%**

b) Top 5 vehicle models with highest number of complaints
   1. RAV: 2499 complaints
   2. CAMRY: 2150 complaints
   3. HIGHLANDER: 1538 complaints
   4. TACOMA: 1325 complaints
   5. PRIUS: 1303 complaints
Note: Above data just gives the details of the number of complaints. Doesn't necessarily say which model is bad as the values are not normalised

c) Complaints mentioning Crash, Fire, Injury, Death, Medical attention required and police reported
   1. Crash: 1403 complaints
   2. Fire: 307 complaints
   3. Injury: 609 complaints
   4. Deaths: 21 complaints
   5. Medical Attention required: 653 complaints
   6. Police reported: 954 complaints

d) Highest number of complaints were regisetered in the following cities
  1. California: 2766 complaints
  2. Florida: 1488 complaints
  3. Texas: 1450 complaints


<h1>Analysing complaints which reported deaths</h1>

 - Between year 2020 and 2024, 21 complaints involving deaths are reported
 - CAMRY, COROLLA and 4RUNNER are the major models
 - Using Large Language Model (Mistral), the main systems responisble for accidents were extracted from the CDESCR field
 - From the description, Large Language Model has identified the following systems
     1. **Air Bag Malfunctioning**
     2. **Steering System Failure**
     3. **Battery and Electrical System Failure**

<h1>Analysing complaints which reported Fire Incident</h1>

 - Between year 2020 and 2024, 307 complaints involving deaths are reported
 - RAV4, CAMRY and HIGHLANDER are the major models
 - Using Large Language Model (Mistral), the main systems responisble for fire incidents were extracted from the CDESCR field
 - From the description, Large Language Model has identified the following systems
     1. **Electrical System Failure**
     2. **Component Overheating**
     3. **Battery Malfunction & Explosion**
 

Toyota Safety Complaint Analysis
This project focuses on analyzing safety complaints registered against Toyota vehicles using data sourced from the National Highway Traffic Safety Administration (NHTSA). The goal is to identify key trends, critical issues, and system-level failures that may compromise vehicle safety.

📊 Key Insights from the Analysis
🗓️ Number of Complaints by Year (2020–2024)
Year	Number of Complaints
2020	4,485
2021	3,223
2022	2,735
2023	3,079
2024	4,092
Insight: From 2022 onwards, there has been an average year-over-year increase of approximately 22% in the number of complaints.

🚗 Top 5 Toyota Models with the Most Complaints
Rank	Model	Complaints
1	RAV4	2,499
2	Camry	2,150
3	Highlander	1,538
4	Tacoma	1,325
5	Prius	1,303
⚠️ Note: This data reflects the total number of complaints and is not normalized by the number of vehicles sold. It doesn't necessarily indicate which model is less reliable.

⚠️ Complaints Involving Critical Incidents
Incident Type	Complaints
Crash	1,403
Fire	307
Injury	609
Death	21
Medical Attention Required	653
Police Report Filed	954
📍 States with the Highest Number of Complaints
Rank	State	Complaints
1	California	2,766
2	Florida	1,488
3	Texas	1,450
🔍 Deep Dive: Complaints Involving Deaths
A total of 21 complaints involving fatalities were registered between 2020 and 2024.

The models most frequently mentioned were Camry, Corolla, and 4Runner.

Using the Mistral Large Language Model, key vehicle systems responsible for the incidents were extracted from the complaint descriptions (CDESCR field).

Identified systems linked to fatal outcomes:

Airbag Malfunction

Steering System Failure

Battery and Electrical System Failure

🔥 Deep Dive: Complaints Involving Fire Incidents
A total of 307 fire-related complaints were recorded from 2020 to 2024.

Models most frequently involved were RAV4, Camry, and Highlander.

The Mistral LLM was used to extract system-level causes from the complaint text.

Commonly identified causes of fire incidents:

Electrical System Failure

Component Overheating

Battery Malfunction and Explosion
   
