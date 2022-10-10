# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt


def grad_desc_f1(print_iters_flag=True):
    print("GRADIENT_DESCENT_TESTING_FUNCTION_1")

    function1 = lambda x: 10 * x**4 + 3 * x**3 - 30 * x**2 + 10 * x
    gradient_function1 = lambda x: 40 * x**3 + 9 * x**2 - 60 * x + 10

    # START POINT
    cur_x = -5

    # STEP RATE
    rate = 0.00001

    # PRECISION
    precision = 0.00000001

    # NUMBER OF ITERATIONS
    max_iters = 10000

    previous_step_size = 1
    iters = 0

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x

        cur_x = cur_x - rate * gradient_function1(prev_x)

        previous_step_size = abs(cur_x - prev_x)

        iters += 1

        if(print_iters_flag):
            print("Iteration", iters, "\nX value is", cur_x)
    
    print("The local minimum:", function1(cur_x), "occurs at:", cur_x)

    #FUNCTION_1_1: The local minimum: -42.627678775749324 occurs at: -1.4123770904085056 (cur_x=-5, rate=0.00001, precision=0.00000001)
    #FUNCTION_1_2: The local minimum: -7.0063217046035575 occurs at: 1.0126646619672222 (cur_x=2, rate=0.00001, precision=0.00000001)


def grad_desc_f2(print_iters_flag=True):
    print("GRADIENT_DESCENT_TESTING_FUNCTION_2")

    function2 = lambda x1, x2: 10 * x2**4 + 10 * x1**4 + 3 * x1**3 - 30 * x1**2 + 10 * x1
    gradient_function2 = [lambda x1: 40 * x1**3 + 9 * x1**2 - 60 * x1 + 10, lambda x2: 40 * x2**3]

    # START POINT
    cur_x1 = -5
    cur_x2 = -5

    # STEP RATE
    rate = 0.001

    # PRECISION
    precision = 0.0000001

    # NUMBER OF ITERATIONS
    max_iters = 10000

    previous_step_size = 1
    iters = 0

    while previous_step_size > precision and iters < max_iters:
        prev_x1 = cur_x1
        prev_x2 = cur_x2

        cur_x1 = cur_x1 - rate * gradient_function2[0](prev_x1)
        cur_x2 = cur_x2 - rate * gradient_function2[1](prev_x2)

        previous_step_size = ((cur_x1 - prev_x1)**2 + (cur_x2 - prev_x2)**2)**(0.5)

        iters += 1

        if(print_iters_flag):
            print("Iteration", iters, "\nX1 value is", cur_x1, "\nX2 value is", cur_x2) #Print iterations
    
    print("The local minimum:", function2(cur_x1, cur_x2), "occurs at:", f'({cur_x1}, {cur_x2})')

    #FUNCTION_2_1: The local minimum: -42.627678778956444 occurs at: (-1.412370090969678, 0.0) (cur_x1=-5, cur_x2=-5 rate=0.001, precision=0.0000001)
    #FUNCTION_2_2: The local minimum: -7.0063217046035575 occurs at: 1.0126646619672222 (cur_x=2, rate=0.00001, precision=0.00000001)

if __name__ == "__main__":
    grad_desc_f1(False)
    grad_desc_f2(False)