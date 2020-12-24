def main():
    exec('x = 1',globals(),locals())
    print(x)


if __name__ == '__main__':  
     for i in range(100):
            exec('x%s = %s'%(i+1))
    print(1)
