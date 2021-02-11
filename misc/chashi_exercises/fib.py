"""
silly project
-------------
"""
fib_array = [0, 1]
def fib(n):
    if n <= 0:
        return 0
    if n == 1:
        return fib_array[0]
    elif n == 2: 
        return fib_array[1]
    else:
        for i in range(2, n):
            temp = fib_array[1]
            fib_array[1] = fib_array[0] + fib_array[1]
            fib_array[0] = temp
        return fib_array[1]
    

if __name__ == "__main__":
    while(True):
        
        print('Enter a numbers: ')
        n = int(input())
        print(fib(n))
        fib_array = [0,1]
        
        
