def format_number(number):
    # 使用科学记数法格式化数字，保留三位小数
    formatted_number = f"{number:.3e}"
    
    # 拆分成系数和指数部分
    coefficient, exponent = formatted_number.split('e')
    
    # 去掉系数中的多余的0
    coefficient = str(float(coefficient))
    
    # 处理指数部分，去掉前导的正号
    exponent = exponent.lstrip('+0') if exponent[0] != '-' else '-' + exponent[1:].lstrip('0')
    
    # 构造最终的字符串形式
    formatted_string = f"{coefficient}*10^{{{exponent}}}"
    
    return formatted_string

# 测试示例
numbers = [12345, 0.0001251]

for number in numbers:
    print(format_number(number))
