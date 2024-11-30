import os
import re


# 遍历文件夹中的所有txt文件
def process_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        # 只处理txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                # 使用正则表达式在开头的两个 "0" 之间插入空格
                new_line = re.sub(r'(^0)(0)', r'\1 \2', line)
                new_lines.append(new_line)

            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.writelines(new_lines)


# 调用函数，传入你要处理的文件夹路径
folder_path = 'val'  # 替换成你的文件夹路径
process_txt_files(folder_path)
