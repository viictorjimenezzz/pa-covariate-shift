from typing import Optional
import seaborn as sns
import matplotlib.font_manager as fm
import functools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from src.plot.dg import LIST_PASPECIFIC_METRICS, COLORS_DICT, METRIC_DICT, LABELS_DICT

import matplotlib.lines as mlines

class ErrorBarHandler(object):
    def __init__(self, capsize=5):
        self.capsize = capsize

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        # Create the vertical line
        line_top = mlines.Line2D([x0+width/2, x0+width/2], [y0, y0+height - 1], color='black')
        line_bottom = mlines.Line2D([x0+width/2, x0+width/2], [y0, y0-height + 1], color='black')
        
        # Create the horizontal cap
        cap = mlines.Line2D([x0+width/2-self.capsize/2 - 1, x0+width/2+self.capsize/2], 
                            [y0+height, y0+height], color='black', lw=1)
        
        handlebox.add_artist(line_top)
        handlebox.add_artist(cap)
        handlebox.add_artist(line_bottom)
        return line_top
    
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SideBySideHandler:
    def __init__(self, pad=0.2):
        self.pad = pad

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        # Calculate widths for each patch
        total_width = width / (1 + self.pad)
        individual_width = total_width / 2
        
        # Create and position the patches
        patch1 = mpatches.Rectangle([x0, y0], individual_width, height, 
                                    facecolor=orig_handle[0].get_facecolor(),
                                    edgecolor=orig_handle[0].get_edgecolor(),
                                    hatch=orig_handle[0].get_hatch())
        
        patch2 = mpatches.Rectangle([x0 + individual_width + self.pad * individual_width, y0], 
                                    individual_width, height,
                                    facecolor=orig_handle[1].get_facecolor(),
                                    edgecolor=orig_handle[1].get_edgecolor(),
                                    alpha=orig_handle[1].get_alpha())
        
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        
        return handlebox

def get_nth_elements(data_dict: dict, n: int):
    nth_elements = {}
    for key, value in data_dict.items():
        if len(value) > n:
            nth_elements[key] = value[n]
        else:
            nth_elements[key] = None  # or handle as needed, e.g., raise an error, use a default value, etc.
    return nth_elements
    

def extract_names(run_string):
    # Define the regular expression pattern
    pattern = r"mod=(?P<mod>\w+)_opt=(?P<opt>\w+)_lr=(?P<lr>[0-9.]+)"
    
    # Match the pattern with the input string
    match = re.match(pattern, run_string)
    
    if match:
        # Extract the named groups
        mod = match.group('mod')
        opt = match.group('opt')
        lr = match.group('lr')
        return mod, opt, lr
    else:
        return None


def generate_latex_table(df, optimizer: str, lr: float, main_factor: str, n_pair: bool):
    # Get the dataset names (e.g., pos_zero_npair, pos_idval_npair, etc.)
    dataset_names = df.columns[2:-1].unique()
    dataset_names_parsed = [dataset_name_parser(ds_name) for ds_name in dataset_names]
    # Get the model names (e.g., erm, irm, etc.)
    model_names = df['model'].unique()
    model_names_parsed = [MODEL_NAMES[model_name] for model_name in model_names]

    # Variable to track if significant improvement was observed in AFR$_P$
    significant_improvement_observed = False
    
    # Start building the LaTeX table
    latex_code = "\\begin{table}[H]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l|cl|cl|cl|cl|cl|cl}\n"

    # Header for shift values
    latex_code += "\\multirow{2}{*}{} & " + " & ".join(
        [
        f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{shift}}}}}" if shift < 5 else f"\\multicolumn{{2}}{{c}}{{\\textbf{{{shift}}}}}"
        for shift in range(6)
    ]
    ) + " \\\\\n"

    # Select the indexes of the best accuracies:    
    list_max_acc, list_max_pa = [], []
    for model in model_names:
        accs_dataset = np.zeros((len(dataset_names), 6))
        for idataset, dataset in enumerate(dataset_names):
            for shift in range(6):
                row = df[(df['shift'] == shift) & (df['model'] == model)]
                if not row.empty:
                    acc =  100.0*row[dataset][row['metric'] == 'acc'].values[0]
                else:
                    acc= 0.0
                accs_dataset[idataset, shift] = acc
        list_max_acc.append(np.argmax(accs_dataset, axis=0))
    
    
    # Iterate over each model
    for imodel, (model, model_parsed) in enumerate(zip(model_names, model_names_parsed)):
        # Header for metrics (Acc., PA)
        latex_code += f"\\textbf{{{model_parsed}}} & " + " & ".join(
            # [r"\textbf{Acc.} & \textbf{PA}" for _ in range(6)]
            [r"Acc. & $\Delta_{\operatorname{PA}}$" for _ in range(6)]
        ) + " \\\\\n"
        latex_code += "\\midrule\n"

        # Iterate over each dataset
        for idataset, (dataset, dataset_parsed) in enumerate(zip(dataset_names, dataset_names_parsed)):
            latex_code += f"{dataset_parsed}"
            # latex_code += f"\\textbf{{{dataset_parsed}}}"
            for shift in range(6):
                # Filter the DataFrame for the current dataset, model, and shift
                row = df[(df['shift'] == shift) & (df['model'] == model)]
                if not row.empty:
                    # Extract the metric values for the current dataset, model, and shift
                    acc = 100.0*row[dataset][row['metric'] == 'acc'].values[0]
                    pa = 100.0*row[dataset][row['metric'] == 'pa'].values[0]

                    # Replace 0.000 with a dash "-"
                    acc_str = f"\\textbf{{{acc:.1f}}}" if idataset == list_max_acc[imodel][shift] else f"{acc:.1f}"
                    if float(f"{pa:.1f}") > 0:
                        pa_str_in = f"\Plus {abs(pa):.1f}"
                        pa_str = f"{{\\color{{tab:green}}  \\textbf{{{pa_str_in}}}}}"
                    elif float(f"{pa:.1f}") < 0:
                        pa_str_in = f"\Minus {abs(pa):.1f}"
                        pa_str = f"{{\\color{{tab:red}} \\textbf{{{pa_str_in}}}}}"
                    else:
                        pa_str = r"\PlusMinus 0.01" #"0.0"
                    
                    latex_code += f" & {acc_str} & {pa_str}"
                else:
                    latex_code += " & - & -"  # Placeholder if there's no data
            latex_code += " \\\\\n"


        if imodel < len(model_names)-1:
            latex_code += "\\midrule\n\\addlinespace\n\\addlinespace\n"
    
    latex_code += "\\bottomrule\n\\end{tabular}%\n}\n"

    # Add caption with significant improvement information
    caption_text = f"REMOVEopt={optimizer}-lr={lr}-mf={main_factor}-npair={n_pair}REMOVE Test performance on increasingly shifted datasets for models selected during ERM and IRM procedures. Different validation datasets are used, and the selection capabilities of PA and validation accuracy are compared."
    latex_code += f"\\caption{{{caption_text}}}\n\\label{{tab:label}}\n\\end" + "{" + "table}"
    
    return latex_code