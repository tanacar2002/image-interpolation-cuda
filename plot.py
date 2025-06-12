import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import sys

def calcSpeedup(tested, ref):
    return np.mean(((ref-tested)/ref)*100)

PLOT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

X = np.array([1, 4, 8, 16, 32, 64])
X = np.array([1, 12, 24, 36, 48, 60, 72, 84, 96, 108])[1:]
Y = np.array([0,1,2,3])
# in ms
cpu = np.array([[[7.3177, 31.0424, 85.0074, 336.95], [30.1391, 124.0431, 335.7708, 1365.171 ]], [[160.3499, 672.0165, 1781.4106, 7254.1974], [646.2925, 2680.0449, 7298.527 , 29114.6187]]])
cpu_a = np.array([[[8.4436, 31.4524, 82.0809, 255.6007], [27.9636, 124.5262, 333.9927, 1040.4358]], [[139.0415, 667.4664, 1760.0543, 5633.5761], [544.7294, 2720.24, 7201.0816, 22458.2821]]])
cpu_cv = np.array([[[23.33780000000000000000, 26.98350000000000000000, 37.13320000000000000000, 137.02880000000000000000, ], [59.14210000000000000000, 65.39070000000000000000, 84.83770000000000000000, 467.11920000000000000000, ], ], [[252.02700000000000000000, 329.07950000000000000000, 443.26230000000000000000, 2874.95660000000000000000, ], [902.75670000000000000000, 1124.31390000000000000000, 1734.38700000000000000000, 11260.76090000000000000000, ], ], ])
gpu_cv = np.array([[[2.43480000000000000000, 2.48180000000000000000, 2.85360000000000000000, 0., ], [2.58860000000000000000, 2.70550000000000000000, 4.21210000000000000000, 0., ], ], [[4.07010000000000000000, 4.22190000000000000000, 12.83650000000000000000, 0., ], [7.71730000000000000000, 8.72720000000000000000, 39.57950000000000000000, 0., ], ], ])
gpu = np.array([[[.12082560000000000000, .15752960000000000000, .44689920000000000000, 3.40050240000000000000, ], [.43526400000000000000, .58656640000000000000, 1.70827840000000000000, 13.38314240000000000000, ], ], [[2.95936640000000000000, 3.53518400000000000000, 9.14220480000000000000, 64.90906870000000000000, ], [11.27771520000000000000, 13.79346560000000000000, 35.74512290000000000000, 243.75291430000000000000, ], ], ])
gpu_v2 = np.array([[[.06938240000000000000, .08596160000000000000, .16021440000000000000, 1.29010880000000000000, ], [.25827840000000000000, .41662080000000000000, .41542080000000000000, 4.18453440000000000000, ], ], [[1.82237120000000000000, 1.82526400000000000000, 2.60867840000000000000, 25.99085090000000000000, ], [6.31655040000000000000, 8.25413440000000000000, 8.28849280000000000000, 86.77891910000000000000, ], ], ])
gpu_tex = np.array([[[.06251520000000000000, .07089600000000000000, .28689600000000000000, .95105920000000000000, ], [.26307840000000000000, .26816000000000000000, .95941120000000000000, 3.53894720000000000000, ], ], [[1.78369920000000000000, 1.81370880000000000000, 4.98250880000000000000, 19.14055380000000000000, ], [6.19697280000000000000, 6.20457600000000000000, 19.38465610000000000000, 67.01672020000000000000, ], ], ])
gpu_tex_v2 = np.array([[[.11790400000000000000, .09510400000000000000, .13076480000000000000, 1.20775360000000000000, ], [.28434240000000000000, .27428800000000000000, .41453120000000000000, 3.85667520000000000000, ], ], [[1.79320640000000000000, 1.81896000000000000000, 2.09313600000000000000, 24.55291520000000000000, ], [6.23064960000000000000, 6.21955840000000000000, 8.24146560000000000000, 70.20255370000000000000, ], ], ])

cpu_min = np.array([cpu, cpu_a, cpu_cv])
cpu_min = np.min(cpu_min, axis=(0))

for i in range(2):
    for f in range(2):
        for alg in range(4):
            print(*[round(imp[i,f,alg],4) for imp in [cpu_cv, cpu, cpu_a, gpu_cv, gpu, gpu_v2, gpu_tex, gpu_tex_v2]], sep=" & ", end=" \\\\\n")

for ref in [cpu_min, gpu, gpu_v2, gpu_tex, gpu_tex_v2]:
    for test in [cpu_min, gpu, gpu_v2, gpu_tex, gpu_tex_v2]:
        print(round(calcSpeedup(test,ref),2),end=" & ")
    print("")



exit()

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