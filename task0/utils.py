import pandas as pd

def save_solution(csv_file,pred_prob):
	with open(csv_file, 'w') as csv:
		df = pd.DataFrame.from_dict({'Id':range(len(pred_prob)),'y': pred_prob})
		df.to_csv(csv,index = False)