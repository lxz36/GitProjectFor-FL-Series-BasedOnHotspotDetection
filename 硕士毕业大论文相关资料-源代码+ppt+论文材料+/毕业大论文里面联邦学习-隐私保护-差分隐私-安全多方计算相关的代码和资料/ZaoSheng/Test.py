
# import os  #删除文件
# os.remove('test1.txt')

# f = open('test.txt')

# text = f.read()#读入全部
# print(text)

# f = open('test.txt')
# lines = f.readlines()#读入每行数据
# print(lines)


# f = open('test.txt')  #
# for line in f:         #顺序读入每行，事实上，我们可以将 f 放在一个循环中，得到它每一行的内容：
#     print(line)




# f.close()#使用完文件后需要关闭


#使用 w 模式时，如果文件不存在会被创建，我们可以查看是否真的写入成功：；如果文件已经存在， w 模式会覆盖之前写的所有内容：
# f = open('myfile.txt', 'w')
# f.write('another hello world')
# f.close()
# print(open('myfile.txt').read())

#除了写入模式，还有追加模式 a ，追加模式不会覆盖之前已经写入的内容，而是在之后继续写入：写入结束之后一定要将文件关闭，否则可能出现内容没有完全写入文件中的情况。
# f = open('myfile.txt', 'a')
# f.write('... and more')
# f.close()
# print(open('myfile.txt').read())

#还可以使用读写模式 w+：这里 f.seek(6) 移动到文件的第6个字符处，然后 f.read() 读出剩下的内容。
# f = open('myfile.txt', 'w+')
# f.write('hello world!')
# f.seek(6)
# print(f.read())
# f.close()





