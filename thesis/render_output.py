import matplotlib.pyplot as plt

def create_figure(clf,X_train, y_test,y_pred, path):


    importances = clf.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.barh(X_train.columns, importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    # --- Save figure ---
    plt.savefig(path, dpi=300)
    # create_table_result(y_test,y_pred,param1,param2,2 )
    
def create_table_result(report_dict,min_lexical_sim,min_cosine_sim,param1, param2, path ):

    latex_lines = []
    latex_lines.append("\\begin{tabular}{ |p{2cm}||p{2cm}|p{2cm}|p{2cm}|p{2cm}|  }")
    latex_lines.append(" \\hline")


    table_title = f"{param2} {param1}"
    latex_lines.append(f" \\multicolumn{{5}}{{|c|}}{{{table_title}}} \\\\")
    latex_lines.append(" \\hline")

    latex_lines.append("  & Precision & Recall & F1 score & support\\\\")
    latex_lines.append(" \\hline")

    # --- Add class 0 ---
    cls_0 = report_dict["0"]
    latex_lines.append(
        f"0   &   {cls_0['precision']:.2f}  &    {cls_0['recall']:.2f}   &   {cls_0['f1-score']:.2f}    &   {int(cls_0['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add class 1 ---
    cls_1 = report_dict["1"]
    latex_lines.append(
        f"1   &   {cls_1['precision']:.2f}  &    {cls_1['recall']:.2f}   &   {cls_1['f1-score']:.2f}    &   {int(cls_1['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add accuracy ---
    accuracy = report_dict["accuracy"]
    total_support = int(cls_0["support"] + cls_1["support"])
    latex_lines.append(
        f"accuracy & & &    {accuracy:.2f}    &   {total_support}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add macro avg ---
    macro_avg = report_dict["macro avg"]
    latex_lines.append(
        f"macro-avg &   {macro_avg['precision']:.2f}   &   {macro_avg['recall']:.2f}  &    {macro_avg['f1-score']:.2f}    &   {int(macro_avg['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add weighted avg ---
    weighted_avg = report_dict["weighted avg"]
    latex_lines.append(
        f"weighted avg &   {weighted_avg['precision']:.2f}   &   {weighted_avg['recall']:.2f}  &    {weighted_avg['f1-score']:.2f}    &   {int(weighted_avg['support'])}\\\\"
    )
    latex_lines.append(" \\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\\\")
    latex_lines.append("\\\\")

    # --- Append to text file ---
    with open(path, "a") as f:  
        f.write("\n".join(latex_lines))
        f.write("\n\n")  # Add some spacing between tables