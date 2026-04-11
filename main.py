import ForecastProcessor as forecastprocessor
import ErrorAnalyzer as erroranalyzer
import MAPECalculator as mapecalculator


file = "forecast.csv"


# 1) ForecastProcessor

fp = forecastprocessor.ForecastProcessor(file)
fp.create_file()
fp.load_data()

y_true, y_pred = fp.filter_non_zero()

print("Filtered data:")
print(y_true)
print(y_pred)



# 2) ErrorAnalyzer (NumPy)

ea = erroranalyzer.ErrorAnalyzer(y_true, y_pred)
ea.run()


# 3) MAPECalculator (Pandas)

mc = mapecalculator.MAPECalculator(file)
mape = mc.run()

print("\nFINAL MAPE:", mape)