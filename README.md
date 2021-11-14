# Self_Build_LSTM
<<<<<<< HEAD
 这是存放自然语言处理课程期末作业：“自己搭建LSTM结点”,的repository
 
## **做以下一些说明:**

## 1、各文件（文件夹）的含义：
    |序号|文件（文件夹）名|类型|内容|
    | :--------:   | :-----:  | :----: |
    |(1)|实验报告-黄元通.docx|Word文件|我的实验报告|
    |(2)|实验报告-黄元通.pdf|pdf文件|pdf版实验报告，内容完全相同|
    |(3)|Code|文件夹|运行实验所需的所有源文件，其中LSTM.py为自己搭建的LSTM类|
    |(4)|LSTM.py|文件|备份，与Code文件下LSTM.py相同|
    |(5)|README.md|文件|仓库说明|

## 2、自己搭建的LSTM类所在文件：“LSTM.py”，该文件可在以下2个位置找到：
   Self_Build_LSTM/LSTM.py \n
   Self_Build_LSTM/Code/LSTM.py \n
  **两个文件完全相同**
   
## 3、在程序中使用自定义LSTM的方法：
    (1) 需要添加语句“from LSTM import LSTM”以使用自定义LSTM
    (2) 使用如下语句进行使用，其中input_size、hidden_size为必须传入的形参，num_layers可不传入，默认值为1:
            'LSTM(input_size=emb_size, hidden_size=n_hidden, num_layers=2)'
    (3) 使用如下语句进行调用，其中从外界指定state初始值的调用其X与(hidden_state, cell_state)可在不同运行设备上，会自动转为在X所在设备:
            使用默认state：
                'outputs, (h_n, c_n) = self.LSTM(X)'
            指定初始state：
                'outputs, (h_n, c_n) = self.LSTM(X, (hidden_state, cell_state))'