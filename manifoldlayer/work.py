input_number = []
for i in range(5):
    input_number.append(input())

# number 1
print('1. ' + str(len(input_number[0]))) # 注意输出字符串格式， 强制转换

# number 2
sum = 0
s = input_number[1]
for alpha in s:
    sum = sum + int(alpha)
print('2. ' + str(sum))

# number 3
num = input_number[2]
length = len(input_number[2])//2  # length 先计算一共有几个奇数位， 数学归纳可得前面表达式
sum = 0
for i in range(length):
    sum = sum + int(num[2*i]) # 2×i 为从左数开始计算的奇数位的序号， 同为数学归纳得。 如果从右开始算奇数位，则序号为 length - 2*i
print('3. ' + str(sum))

# number 4
print('4. ' + str(input_number[3].count('4'))) # 直接调用 python 内置 字符串计数方法 count

# number 5
s = input_number[4]
middle = s[(len(s)-1)//2] # 数学归纳总结， 得中间数的序号是 字符串长度减1的差，再除以2取整， python 除法取整 表达式为 //
print('5. ' + middle)
