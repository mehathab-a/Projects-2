
# Yield Prediction and Forecasting 

#### Forecasting is achieved using Time Lagged Shift Sequencing
ie : every n number of historical data is considered to generated yield for the next year


Uses 
Python=3.8
tensorflow = 2.X.X

### For Yield Prediction :
#### Yield(hg/ha) and Area Harvested (ha) is considered as input from user
#### Weather parameters :
      1. Precipitation
      2. Solar Radiation
      3. Snow Water Equivalent
      4. Max Temp
      5. Min Temp
      6. Vapour Pressure
      Dataset obtained from Daymet Single Pixel API



#### Model used is LSTM for Yield Prediction and forecasting is achieved by Time Lagged Shift Sequencing
##### Model Accurately can be used to forecast 1+ year's yield
