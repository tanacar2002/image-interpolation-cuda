import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import sys

PLOT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

X = np.array([1, 4, 8, 16, 32, 64])
X = np.array([1, 12, 24, 36, 48, 60, 72, 84, 96, 108])[1:]
Y = np.array([0,1,2,3])
gpu = np.array([[0.09392, 0.0891232, 0.0859136, 0.0864736, 0.0879648, 0.0922752, ], [0.119827, 0.0853728, 0.0789984, 0.0766688, 0.0744704, 0.0819264, ], [0.214867, 0.138563, 0.124854, 0.122822, 0.118841, 0.121581, ], [1.3397, 0.758857, 0.664182, 0.619295, 0.59763, 0.586341, ], ])
gpu_v2 = np.array([[0.186316, 0.0960128, 0.093568, 0.0947648, 0.0939936, 0.0948672, 0.0913344, 0.097408, 0.0920096, 0.0950272, ], [0.438717, 0.0769856, 0.063664, 0.0583648, 0.0633184, 0.062144, 0.0593856, 0.0641056, 0.0627168, 0.0640672, ], [0.91797, 0.119939, 0.0828192, 0.070864, 0.0828288, 0.0771264, 0.0754656, 0.0784192, 0.075888, 0.0760992, ], [7.2727, 0.632371, 0.366086, 0.291203, 0.352269, 0.316176, 0.297114, 0.322506, 0.304998, 0.299516, ], ])
gpu_single = np.array([[0.0713152, 0.0603584, 0.0581984, 0.0607808, 0.0641792, 0.0712672, ], [0.092128, 0.0530144, 0.0459904, 0.0426112, 0.0423232, 0.0509216, ], [0.169281, 0.0779968, 0.0648, 0.0583776, 0.0579616, 0.06272, ], [1.04004, 0.372512, 0.259344, 0.208806, 0.183139, 0.177869, ], ])
gpu_single_v2 = np.array([[0.140858, 0.063328, 0.0594208, 0.0574496, 0.0620576, 0.0596224, 0.0588768, 0.0595904, 0.0603392, 0.0625664, ], [0.367322, 0.0549184, 0.0444384, 0.0396224, 0.0436736, 0.0423136, 0.0418656, 0.0442368, 0.0437568, 0.0450816, ], [0.787513, 0.0968768, 0.0616928, 0.0504384, 0.0580416, 0.0552704, 0.0531232, 0.0571104, 0.0554208, 0.0538688, ], [5.52637, 0.499813, 0.271325, 0.195247, 0.251426, 0.216222, 0.199095, 0.221381, 0.208224, 0.194429, ], ])
cpu = np.array([1.1149, 3.3369, 7.1274, 52.6891])
thrust = np.array([1.4220992, 1.7801696, 2.4807872, 3.4736832])
thrust_v2 = np.array([0.8808128, 0.7472864, 0.7358336, 0.9368832])

print(((thrust_v2-np.min(gpu_single_v2, axis=1))/thrust_v2)*100)
print(np.mean(((thrust_v2-np.min(gpu_single_v2, axis=1))/thrust_v2)*100))

if PLOT == 0:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    mX, mY = np.meshgrid(X, Y)
    # Plot the surface.
    surf = ax.plot_surface(mX, mY, gpu_single_v2[:,1:], cmap=cm.coolwarm, linewidth=0)#, antialiased=False)

    # A StrMethodFormatter is used automatically
    ax.set_xticks(X, X)
    ax.set_yticks([0, 1, 2, 3], ["640x426", "1280x843", "1920x1280", "5184x3456"])
    ax.zaxis.set_major_formatter('{x:.01f}')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Number of Blocks (1024 TpB)")
    ax.set_ylabel("Input Image Size")
    ax.set_zlabel("Execution Time (ms)")

elif PLOT == 1:
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(Y, cpu, label= "CPU")

    ax.set_xticks(Y, ["640x426", "1280x843", "1920x1280", "5184x3456"])
    ax.set_xlabel("Input Image Size")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend()
    # plt.savefig(f"cpu.png")

    ax.plot(Y, np.min(gpu_v2, axis=1), label= "GPU Multi Kernel")
    ax.plot(Y, np.min(gpu_single_v2, axis=1), label= "GPU Single Kernel")
    ax.plot(Y, thrust, label= "Thrust v2.0.1")
    ax.plot(Y, thrust_v2, label= "Thrust v2.8.4")

    ax.legend()
    plt.savefig(f"cpu_v_gpu_v_multi.png")

elif PLOT == 2:
    plt.figure()
    fig, ax = plt.subplots()

    ax.plot(X, gpu_v2[-1,1:], label= "GPU")
    ax.plot(X, gpu_single_v2[-1,1:], label= "GPU Single Kernel")

    #ax.set_xticks(Y, ["640x426", "1280x843", "1920x1280", "5184x3456"])
    ax.set_xticks(X, X)
    ax.set_xlabel("Number of Blocks (1024 TpB)")
    ax.set_ylabel("Execution Time (ms)")

    ax.legend()
    plt.savefig(f"gpu_v_single.png")
plt.show()