import pandas as pd

import Classification as cl
import DataProcessing as dp

num_comp = 1
num_img = 10

iterations = 1


def experiment(dataName="FaceImage", height=250, width=250, col='Average'):
    final_results = []
    final_results_reg = []
    final_results_single = []

    for i in range(iterations):
        print("Iteration:", i + 1, "/", iterations)
        print("Generating data...")
        (training_data, testing_data, training_data_single, testing_data_single, testList, dataList, full_len, test_len,
         full_len_single, test_len_single, train, test, protected_ts_race, protected_ts_sex, protected_ts_AB_race,
         protected_ts_AB_sex, protected_ts_AB_race_single, protected_ts_AB_sex_single) = dp.processData(
            h=height, w=width, col=col, num_comp=num_comp, num_img=num_img)
        print("Data generated.")

        # (mse, r2, p_coef, p_value, s_coef, s_value, MI_race, MI_sex, r_sep_race, r_sep_sex, accuracy_r, f1_r,
        #  precision_r, recall_r) = cl.regressionExperiment(
        #     train_val=train,
        #     test=test,
        #     comp_test=testing_data,
        #     height=height,
        #     width=width, col=col,
        #     protected_ts_race=protected_ts_race, protected_ts_sex=protected_ts_sex)
        #
        # result_reg = {"Full data size": full_len, "Testing data size": test_len,
        #               "MSE": mse, "R2": r2, "P Coef": p_coef,
        #               "P Value": p_value, "SP Coef": s_coef,
        #               "SP Value": s_value, "MI_race": MI_race, "MI_sex": MI_sex, "R_sep_race": r_sep_race,
        #               "R_sep_sex": r_sep_sex, "Recall": recall_r, "Precision": precision_r,
        #               "F1": f1_r, "Accuracy": accuracy_r}

        # (recall, precision, f1, acc, AOD_race, AOD_sex, spearmanr, sp_pvalue, pearsonr, p_pvalue, MI_encoder_race,
        #  MI_encoder_sex) = cl.comparabilityExperiment(
        #     dataName="FaceImage",
        #     train_val=training_data,
        #     test=testing_data,
        #     testList=testList,
        #     dataList=dataList,
        #     height=height,
        #     width=width,
        #     protected_ts_race=protected_ts_race, protected_ts_sex=protected_ts_sex,
        #     protected_ts_AB_race=protected_ts_AB_race, protected_ts_AB_sex=protected_ts_AB_sex)
        #
        # result = {"Full data size": full_len, "Testing data size": test_len,
        #           "Recall": recall, "Precision": precision, "F1": f1, "Accuracy": acc,
        #           "AOD_race": AOD_race, "AOD_sex": AOD_sex, "Spearman's rank correlation": spearmanr,
        #           "SP value": sp_pvalue,
        #           "Pearson's rank correlation": pearsonr, "P value": p_pvalue,
        #           "MI_encoder_race": MI_encoder_race, "MI_encoder_sex": MI_encoder_sex}

        (recall_single, precision_single, f1_single, acc_single, AOD_race_single, AOD_sex_single, spearmanr_single,
         sp_pvalue_single, pearsonr_single, p_pvalue_single, MI_encoder_race_single,
         MI_encoder_sex_single) = cl.comparabilityExperiment(
            dataName="FaceImage",
            train_val=training_data_single,
            test=testing_data_single,
            testList=testList,
            dataList=dataList,
            height=height,
            width=width,
            protected_ts_race=protected_ts_race, protected_ts_sex=protected_ts_sex,
            protected_ts_AB_race=protected_ts_AB_race_single, protected_ts_AB_sex=protected_ts_AB_sex_single)

        result_single = {"Full data size": full_len_single, "Testing data size": test_len_single,
                         "Recall": recall_single, "Precision": precision_single, "F1": f1_single, "Accuracy": acc_single,
                         "AOD_race": AOD_race_single, "AOD_sex": AOD_sex_single,
                         "Spearman's rank correlation": spearmanr_single,
                         "SP value": sp_pvalue_single,
                         "Pearson's rank correlation": pearsonr_single, "P value": p_pvalue_single,
                         "MI_encoder_race": MI_encoder_race_single, "MI_encoder_sex": MI_encoder_sex_single}

        # final_results.append(result)
        final_results_reg.append(result_reg)
        final_results_single.append(result_single)

    final_results = pd.DataFrame(final_results)
    final_results_reg = pd.DataFrame(final_results_reg)
    final_results_single = pd.DataFrame(final_results_single)

    print("\n*******************************************")
    print(dataName)
    print("*******************************************\n")

    # final_results.to_csv(
    #     "../../Results/" + dataName + " Shared Encoder_" + col + "_" + str(num_img) + "_" + str(num_comp) + ".csv",
    #     index=False)
    final_results_reg.to_csv(
        "../../Results/" + dataName + " Reg_" + col + "_" + str(num_img) + "_" + str(num_comp) + ".csv", index=False)
    final_results_single.to_csv(
        "../../Results/" + dataName + " Shared Encoder Single_" + col + "_" + str(num_img) + "_" + str(
            num_comp) + ".csv", index=False)


experiment(dataName="FaceImage", col='3', height=250, width=250)
