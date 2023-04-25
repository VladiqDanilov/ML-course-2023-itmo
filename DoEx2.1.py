import pandas as pd
df = pd.read_csv(r'ROSSTAT_SALARY_RU.csv', encoding="UTF-8")
df = df.set_index("region_name").drop(
    ['Томская область',  
     'Самарская область']
     , axis = 0)
#print()
var_series = df['salary'].sort_values().reset_index(drop=True)

print(f"X[21] = {var_series[20]}; X[37] = {var_series[36]}; X[41] = {var_series[40]}")
print(f"Выборочное среднее: {round(var_series.mean(), 2)}")
print(f"Медиана: {var_series.median() }")