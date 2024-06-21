import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import pandas as pd
import math

ops = {
    'Relu': '400de06c <_ZN6tflite12_GLOBAL__N_18ReluEvalEP13TfLiteContextP10TfLiteNode>:',
    'Conv2D': '400de528 <_ZN6tflite12_GLOBAL__N_18ConvEvalEP13TfLiteContextP10TfLiteNode>:',
    'Dequantize': '400dfb0c <_ZN6tflite14DequantizeEvalEP13TfLiteContextP10TfLiteNode>:',
    'FullyConnected': '400e0220 <_ZN6tflite12_GLOBAL__N_118FullyConnectedEvalEP13TfLiteContextP10TfLiteNode>:',
    'Logistic': '400e104c <_ZN6tflite12_GLOBAL__N_112LogisticEvalEP13TfLiteContextP10TfLiteNode>:',
    'Maxpool': '400e1508 <_ZN6tflite12_GLOBAL__N_17MaxEvalEP13TfLiteContextP10TfLiteNode>:',
    'Quantize': '400dd7fc <_ZN6tflite21EvalQuantizeReferenceEP13TfLiteContextP10TfLiteNode>:',
    'Reshape': '400e1c6c <_ZN6tflite12_GLOBAL__N_120EvalReshapeReferenceEP13TfLiteContextP10TfLiteNode>:'
}
asm_path = 'person_detection.S'
curr_op = None
op_count = {
    'Relu': {},
    'Conv2D': {},
    'Dequantize': {},
    'FullyConnected': {},
    'Logistic': {},
    'Maxpool': {},
    'Quantize': {},
    'Reshape': {}
}

with open(asm_path, 'r') as f:
    asm = f.readlines()

for line in asm:
    if line == '	...\n' or line == '\n':
        curr_op = None
        continue
    line = line.strip()
    if line in ops.values():
        curr_op = list(ops.keys())[list(ops.values()).index(line)]
        continue
    if curr_op:
        instruction = line.split()[2]
        if instruction not in op_count[curr_op]:
            op_count[curr_op][instruction] = 1
        else:
            op_count[curr_op][instruction] += 1

for op in op_count:
    keys = list(op_count[op].keys())
    values = list(op_count[op].values())

    df = pd.DataFrame({'Instruction': keys, 'Count': values}).sort_values('Count', ascending=False)

    df_5 = df.head(5)

    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

    labels = np.where(np.isin(keys, df_5['Instruction']), keys, '')
    ax[0].pie(values, labels=labels)
    ax[0].set_title(op)


    half = math.ceil(len(df) / 2)
    df1 = df.iloc[:half]
    df2 = df.iloc[half:]


    for i, df_i in enumerate([df1, df2], start=1):
        ax[i].axis('off')
        table = Table(ax[i], bbox=[0, 0, 1, 1])

        
        for j, column in enumerate(df_i.columns):
            table.add_cell(0, j, 1, 0.1, text=column, facecolor='gray', edgecolor='black', loc='center')

        
        for j, (index, row) in enumerate(df_i.iterrows()):
            for k, cell in enumerate(row):
                table.add_cell(j+1, k, 1, 0.1, text=cell, facecolor='white', edgecolor='black', loc='center')

        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        ax[i].add_table(table)

    plt.show()
