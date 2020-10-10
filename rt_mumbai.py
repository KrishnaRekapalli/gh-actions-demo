import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from rtmoddel.generative import GenerativeModel


india_cases = pd.read_csv("city_metrics.csv")

india_cases_mumbai = india_cases[india_cases["district"]=="Mumbai"]

cols_for_rt = ["date", "delta.tested", "delta.confirmed"]

india_cases_mumbai_rt = india_cases_mumbai[cols_for_rt]

india_cases_mumbai_rt["date"] = pd.to_datetime(india_cases_mumbai_rt["date"])

india_cases_mumbai_rt = india_cases_mumbai_rt.rename(columns={"delta.tested":"total", "delta.confirmed":"positive"})

india_cases_mumbai_rt = india_cases_mumbai_rt.set_index("date")

gm2 = GenerativeModel("Mumbai",  india_cases_mumbai_rt)

gm2.sample()

result_mumbai = summarize_inference_data(gm2.inference_data)

print(result_mumbai.tail(10))



