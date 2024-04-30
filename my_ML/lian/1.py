# 读取情报文件并解密
#这段代码是Python中的一种常见用法，用于打开文件并读取其内容。具体解释如下：

#open("secret.daz", "r"): 打开名为 "secret.daz" 的文件，使用只读模式（"r"）。
#as f: 将打开的文件对象赋值给变量 f，这样我们就可以通过 f 来操作文件。
#f.read(): 读取文件的全部内容。
#.strip(): 删除字符串两端的空格和换行符，以防止不必要的空白字符干扰后续的处理。

with open("secret.daz", "r") as f:
    encrypted_data = f.read().strip()

# 解密情报文件
"""
这段代码用于解密情报文件，它的作用是遍历经过分割的十六进制字符串，将其转换为对应的Unicode字符，并拼接成解密后的文本。
具体解释如下：
decoded_data = "": 创建一个空字符串，用于存储解密后的文本。
for hex_value in encrypted_data.split("XX"):
encrypted_data.split("XX"): 将加密的十六进制字符串按照 "XX" 分割成多个部分，并返回一个包含这些部分的列表。
for hex_value in ...: 遍历这个列表中的每个部分，每个部分代表一个十六进制数字。
if hex_value:: 检查当前的十六进制字符串是否非空。
decoded_data += chr(int(hex_value, 16)):
int(hex_value, 16): 将当前的十六进制字符串 hex_value 转换为对应的十进制整数。
chr(...): 将十进制整数转换为对应的Unicode字符。
+=: 将转换后的Unicode字符拼接到 decoded_data 字符串的末尾。
通过这个循环，我们可以逐个解密每个十六进制数字，并将其转换为对应的Unicode字符，最终得到完整的解密后的文本。
"""

decoded_data = ""
for hex_value in encrypted_data.split("X"):
    if hex_value:
        decoded_data += chr(int(hex_value, 16))

#decoded_data = "".join(chr(int(hex_value, 16)) for hex_value in encrypted_data.split("XX") if hex_value)
# 写入解密后的情报到新文件interpretation.txt
with open("interpretation.txt", "w", encoding="utf-8") as f:
    f.write(decoded_data)

# 计算情报总字数
"""
    在 Python 中，char.strip() 方法会去除字符串两端的空格、制表符和换行符，
如果去除后字符串为空，则表示该字符不是可见字符。
    所以，这行代码会遍历解密后的文本 decoded_data 中的每个字符，检查是否是可见字符，
然后使用生成器表达式（generator expression）计算可见字符的数量，并将其求和。
"""
visible_chars = sum(1 for char in decoded_data if char.strip())

# 生成签名
signature = f"<解密人>1234567<情报总字数>{visible_chars}"

# 在interpretation.txt文件末尾写入1个换行符，再写入签名
with open("interpretation.txt", "a", encoding="utf-8") as f:
    f.write("\n" + signature)
