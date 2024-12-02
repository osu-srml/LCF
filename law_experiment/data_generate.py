import pystan
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path

def get_pystan_train_dic(pandas_df, sense_cols):
    dic_out = {}
    dic_out["N"] = len(pandas_df)
    dic_out["K"] = len(sense_cols)
    dic_out["a"] = np.array(pandas_df[sense_cols])
    dic_out["ugpa"] = list(pandas_df["UGPA"])
    dic_out["lsat"] = list(pandas_df["LSAT"].astype(int))
    dic_out["zfya"] = list(pandas_df["ZFYA"])
    return dic_out
    
def get_pystan_test_dic(fit_extract, test_dic):
    dic_out = {}
    for key in fit_extract.keys():
        if key not in ["sigma_g_Sq_1", "sigma_g_Sq_2", "u", "eta_a_zfya", "lp__"]:
            dic_out[key] = np.mean(fit_extract[key], axis=0)
    
    need_list = ["N", "K", "a", "ugpa", "lsat"]
    for data in need_list:
        dic_out[data] = test_dic[data]
    return dic_out

def get_data_preprocess(seed):
    law_data = pd.read_csv("./data.csv", index_col=0)
    law_data = law_data[["Gender", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Married", "Property_Area"]]
    
    law_data = pd.get_dummies(law_data, columns=["Married"], prefix="", prefix_sep="")
    law_data = pd.get_dummies(law_data, columns=["Property_Area"], prefix="", prefix_sep="")
    
    law_data = law_data.dropna()
    
    law_data.rename(columns={"ApplicantIncome": "UGPA", "CoapplicantIncome": "LSAT", "LoanAmount": "ZFYA"}, inplace=True)
    
    law_data["male"] = law_data["Gender"].map(lambda z: 1 if z == "Male" else 0)
    law_data["female"] = law_data["Gender"].map(lambda z: 1 if z == "Female" else 0)
    
    law_data = law_data.drop(axis=1, columns=["Gender"])
    
    sense_cols = ["No", "Yes", "Rural", "Semiurban", "Urban", 'male','female']
    
    law_train, law_test = train_test_split(law_data, random_state=seed, test_size=0.2)
    
    law_train_dic = get_pystan_train_dic(law_train, sense_cols)
    law_test_dic = get_pystan_train_dic(law_test, sense_cols)
    
    return law_train_dic, law_test_dic

def model_learning(law_train_dic, law_test_dic, seed):
    check_fit = Path("./MC_models/model_fit_{}.pkl".format(seed))
    
    if check_fit.is_file():
        print("File Found: Loading Fitted Training Model Samples...")
        with open("./MC_models/model_fit_{}.pkl".format(seed), "rb") as f:
            post_samps = pickle.load(f)
    else:
        print("File Not Found: Fitting Training Model...\n")
        model = pystan.StanModel(file="./stans/law_school_train.stan")
        print("Finished compiling model!")
        print("keys = {}".format(law_train_dic.keys()))
        fit = model.sampling(data=law_train_dic, iter=1000, chains=1)
        post_samps = fit.extract()
        with open("MC_models/model_fit_{}.pkl".format(seed), "wb") as f:
            pickle.dump(post_samps, f, protocol=-1)
        print("Saved fitted model!")
    
    law_train_dic_final = get_pystan_test_dic(post_samps, law_train_dic)
    law_test_dic_final = get_pystan_test_dic(post_samps, law_test_dic)
    
    check_train = Path("./MC_models/model_fit_train_{}.pkl".format(seed))
    
    if check_train.is_file():
        print("File Found: Loading Test Model with Train Data...")
        with open("MC_models/model_fit_train_{}.pkl".format(seed), "rb") as f:
            fit_train_samps = pickle.load(f)
    else:
        print("File Not Found: Fitting Test Model with Train Data...\n")
        model_train = pystan.StanModel(file="./stans/law_school_only_u.stan")
        fit_train = model_train.sampling(data=law_train_dic_final, iter=2000, chains=1)
        fit_train_samps = fit_train.extract()
        with open("MC_models/model_fit_train_{}.pkl".format(seed), "wb") as f:
            pickle.dump(fit_train_samps, f, protocol=-1)
        print("Saved train samples!")
    
    check_test = Path("./MC_models/model_fit_test_{}.pkl".format(seed))
    
    if check_test.is_file():
        print("File Found: Loading Test Model with Test Data...")
        with open("MC_models/model_fit_test_{}.pkl".format(seed), "rb") as f:
            fit_test_samps = pickle.load(f)
    else:
        print("File Found: Loading Test Model with Test Data...")
        model_test = pystan.StanModel(file="./stans/law_school_only_u.stan")
        fit_test = model_test.sampling(data=law_test_dic_final, iter=2000, chains=1)
        fit_test_samps = fit_test.extract()
        with open("MC_models/model_fit_test_{}.pkl".format(seed), "wb") as f:
            pickle.dump(fit_test_samps, f, protocol=-1)

def data_generation(seed):
    law_train_dic, law_test_dic = get_data_preprocess(seed)
    model_learning(law_train_dic, law_test_dic, seed)
    
    data = {}
    
    with open("./MC_models/model_fit_{}.pkl".format(seed), "rb") as f:
        parameters_samples = pickle.load(f)
        
    data["bG"] = np.mean(parameters_samples["ugpa0"])
    data["wGK"] = np.mean(parameters_samples["eta_u_ugpa"])
    data["wGR"] = np.mean(parameters_samples["eta_a_ugpa"][:, :-2], axis=0)
    data["wGS"] = np.mean(parameters_samples["eta_a_ugpa"][:, -2:], axis=0)
    data["sigma_1"] = np.mean(parameters_samples["sigma_g_Sq_1"])
    data["sigma_2"] = np.mean(parameters_samples["sigma_g_Sq_2"])
    
    data["bL"] = np.mean(parameters_samples["lsat0"])
    data["wLK"] = np.mean(parameters_samples["eta_u_lsat"])
    data["wLR"] = np.mean(parameters_samples["eta_a_lsat"][:, :-2], axis=0)
    data["wLS"] = np.mean(parameters_samples["eta_a_lsat"][:, -2:], axis=0)
    
    data["wFK"] = np.mean(parameters_samples["eta_u_zfya"])
    data["wFR"] = np.mean(parameters_samples["eta_a_zfya"][:, :-2], axis=0)
    data["wFS"] = np.mean(parameters_samples["eta_a_zfya"][:, -2:], axis=0)
    
    data["train_S"] = np.array(law_train_dic["a"][:, -2:])
    data["train_R"] = np.array(law_train_dic["a"][:, :-2])
    data["train_G"] = np.array(law_train_dic["ugpa"])
    data["train_L"] = np.array(law_train_dic["lsat"])
    data["train_F"] = np.array(law_train_dic["zfya"])
    
    data["test_S"] = np.array(law_test_dic["a"][:, -2:])
    data["test_R"] = np.array(law_test_dic["a"][:, :-2])
    data["test_G"] = np.array(law_test_dic["ugpa"])
    data["test_L"] = np.array(law_test_dic["lsat"])
    data["test_F"] = np.array(law_test_dic["zfya"])
    
    with open("MC_models/model_fit_train_{}.pkl".format(seed), "rb") as f:
        train_K = pickle.load(f)
    data["train_K"] = train_K["u"].T
    
    with open("MC_models/model_fit_test_{}.pkl".format(seed), "rb") as f:
        test_K = pickle.load(f)
    data["test_K"] = test_K["u"].T
    
    with open("datas/data_{}.pkl".format(seed), "wb") as f:
        pickle.dump(data, f, protocol=-1)
        
data_generation(seed=42)