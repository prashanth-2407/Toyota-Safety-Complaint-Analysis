# Toyota-Safety-Complaint-Analysis

In this project, I have analysed the safety complaints registered against Toyota's vehicle which is available in [NHTSA](https://www.nhtsa.gov/).

<h1>Some interesting insights drawn from the analysis</h1>

 - 📅 Number of safety complaints registered against toyota's vehicles from 2020 to 2024
   1. Year 2020: 4485 complaints
   2. Year 2021: 3223 complaints
   3. **Year 2022: 2735 complaints**
   4. **Year 2023: 3079 complaints**
   5. **Year 2024: 4092 complaints**
      
<h3>📈 The average Year-over-year increase in complaints from 2022 onwards is approximately 22%</h3>

 - 🚘 Top 5 vehicle models with highest number of complaints
   1. **RAV: 2499 complaints**
   2. CAMRY: 2150 complaints
   3. HIGHLANDER: 1538 complaints
   4. TACOMA: 1325 complaints
   5. PRIUS: 1303 complaints

⚠️ Note: Above data just gives the details of the number of complaints. Doesn't necessarily say which model is bad as the values are not normalised

 - <h3>🆘 Complaints mentioning Crash, Fire, Injury, Death, Medical attention required and police reported</h3>
   1. 💥 Crash: 1403 complaints
   2. 🔥 Fire: 307 complaints
   3. 🤕 Injury: 609 complaints
   4. ⚰️ **Deaths: 21 complaints**
   5. 🏥 Medical Attention required: 653 complaints
   6. 👮 Police reported: 954 complaints

 - <h3>Highest number of complaints were regisetered in the following cities</h3>
   1. **California: 2766 complaints**
   2. Florida: 1488 complaints
   3. Texas: 1450 complaints


<h3>Analysing complaints which reported deaths</h3>

 - Between year 2020 and 2024, 21 complaints involving deaths are reported
 - CAMRY, COROLLA and 4RUNNER are the major models
 - Using Large Language Model (Mistral), the main systems responisble for accidents were extracted from the CDESCR field
 - From the description, Large Language Model has identified the following systems
     1. 🛡️ **Air Bag Malfunctioning**
     2. 🔄 **Steering System Failure**
     3. 🔋 **Battery and Electrical System Failure**

<h3>Analysing complaints which reported Fire Incident</h3>

 - Between year 2020 and 2024, 307 complaints involving deaths are reported
 - RAV4, CAMRY and HIGHLANDER are the major models
 - Using Large Language Model (Mistral), the main systems responisble for fire incidents were extracted from the CDESCR field
 - From the description, Large Language Model has identified the following systems
     1. ⚡ **Electrical System Failure**
     2. 🌡️ **Component Overheating**
     3. 💥 **Battery Malfunction & Explosion**
 

   
