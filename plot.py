import matplotlib.pyplot as plt
import numpy as np
'''
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
ax1.plot(x,y)
plt.show()

for i in range(len(a)):
    print(i)
    ax.insert(i,"ax"+str(i))

print(tuple(ax))
print(ax[1])

'''
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
df_confusion = [x,y]
ax = []
a = (1,2)
for i in range(len(a)):
    ax.insert(i,"ax"+str(i))
    ax[i]
fig, ax = plt.subplots(1,2,sharey=True)
for idx,df in enumerate(df_confusion):
    ax[idx].plot(x,y)

plt.show()

