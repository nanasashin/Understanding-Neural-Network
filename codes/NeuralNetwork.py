input  = [1, 3, 6, 4]

weights = {
    0: [2.4, 6, 3, 4.6],
    1: [5.0, 10, 9.8, 8.5],
    2: [2.9, 1.6, 5.3, 2.3],
}

bias = [3, 10, 9]

output = []
x = 1

for x in range(0, len(bias)): 
    output.append(input[0] * weights[x][0] + input[1] * weights[x][1] + input[2] * weights[x][2] + input[3] * weights[x][3] + bias[x])

print(output)