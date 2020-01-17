import math
import numpy as np
import matplotlib.pyplot as plt

def hitung_total_dist(rute):
    jum = 0
    for i in range(len(rute)-1):
        jum += arrdist[rute[i]][rute[i+1]]
        # print(i)
    return jum


def hitung_probabilitas(semut, i, arrJalan):
    kota_sekarang = semut[len(semut)-1]
    # kota_sekarang menuju kota i
    # current_city to city-i
    a = (pheromone[kota_sekarang][arrJalan[i]]**alpha) * (visibility[kota_sekarang][arrJalan[i]]**beta)
    b = 0
    for j in arrJalan:
        b += (pheromone[kota_sekarang][j]**alpha) * (visibility[kota_sekarang][j]**beta)
    return a/b

def pilih_kota_selanjutnya(semut, jumlah_kota):
    arrJalan = list(set(np.arange(jumlah_kota)) - set(semut))
    arrProb  = np.zeros(len(arrJalan))
    for i in range(len(arrJalan)) :
        # hitung probabilitas
        # count the probability
        arrProb[i] = hitung_probabilitas(semut, i, arrJalan)
    # print(arrProb)
    kota_next = np.random.choice(arrJalan, 1, p=arrProb)
    return kota_next[0]


def dist(point1, point2):
    # function menghitung distance antara 2 titik
    # distance between 2 point/city function
    distX = (point1[1] - point2[1])**2
    distY = (point1[2] - point2[2])**2
    return math.sqrt(distX + distY)

#-------pengambilan data dari file------------
# data acquisition
with open("swarm096.tsp") as f:
		content = f.read().splitlines()
		cleaned = [x.lstrip() for x in content if x != ""]


swarm96 = []
for i in range(1,len(cleaned)-1) :
    swarm96.append([float(cleaned[i].split()[0]), float(cleaned[i].split()[1]), float(cleaned[i].split()[2])])

swarm96_np = np.array(swarm96)
#---------------------------------------------

matrixjarak96 = np.zeros((96,96))

#inisialisasi jarak full connected
# full connected distance initialization
for i in range(96):
    for j in range(96):
        matrixjarak96[i][j] = dist(swarm96_np[i], swarm96_np[j])



#plt.figure()
#plt.plot(swarm16_np[:,1], swarm16_np[:,2])
#plt.show()

#-------------------------
# arrdist = np.array([[0.1, 9, 1, 9],
#                     [9, 0.1, 9, 9],
#                     [1, 9, 0.1, 9],
#                     [9, 9, 9, 0.1]])

arrdist = np.array(matrixjarak96)

for i in range (96):
    for j in range(96):
        if i == j :
            # if same city
            arrdist[i][j] = 0.1
# distance kota i, i diisi dengan 0.1 untuk mengindari error saat dibagi
# distance for the same city replace with 0.1 to avoid error
visibility = 1 / arrdist

time = 0
NC   = 100
c    = 0.00001

alpha  = 1
beta   = 2
koef   = 0.8

pheromone   = np.ones ((96, 96)) * c
delta_pher  = np.zeros((96, 96))

Q    = 100

n    = 96 # jumlah kota    /  total city
m    = 96 # jumlah semut   / total ant

best_route      = []
# total_dist_best = 0

bool_best_pertama = False
# bool_best_pertama untuk menandakan apakah sudah ada isi dari best distance atau belum
# ^this variable is to check if best distance is already set or not

# pengulangan
# loop
for loop in range(NC):
    # print("ok")
    tabu    = []
    for isi in range(n):
        tabu.append([])
    #-------------------------------------------------
    s = 0
    for i in range(m):
        tabu[i].append(i)

    while len(tabu[0]) < n:
        for k in range(m) :
            kota_selanjutnya = pilih_kota_selanjutnya(tabu[k], 96)
            tabu[k].append(kota_selanjutnya)



    for l in range(m):
        tabu[l].append(tabu[l][0])
    # ---------------------------------------------langkah 4

    # ----inisialisasi jarak best pertama kali---------------
    # initialization of best distance for the first time
    if(not bool_best_pertama):
        best_route = tabu[0]
        bool_best_pertama = True

    # -------------------------------------------------------

    for k in range(m):
        jum = hitung_total_dist(tabu[k])
        # cek jika distance yang didapat lebih baik dari sebelumnya
        # v check if the current distance better than before
        if(hitung_total_dist(best_route) > jum):
            best_route = tabu[k]

        for l in range(len(tabu[k])-1):
            delta_pher[tabu[k][l]][tabu[k][l+1]] += Q/hitung_total_dist(tabu[k])



    # -----------------------------------------langkah 5
    for ii in range(n):
        for jj in range(n):
            if(ii != jj):
                pheromone[ii][jj] = koef * pheromone[ii][jj] + delta_pher[ii][jj]

    delta_pher  = np.zeros((n, n))

    print(loop, " :: " , hitung_total_dist(best_route), best_route)


#----------------visualisasi--------------------
# visualization
plt.axis([-40, 50, -30, 60])
xx = []
yy = []
for i in range(97):
    xx.append(swarm96_np[best_route[i]][1])
    yy.append(swarm96_np[best_route[i]][2])
plt.plot(xx, yy, '-')
plt.show()
#-----------------visualisasi : end--------------
# visualization end

# aa = np.array([1, 0, 3, 2, 1])
# print(hitung_total_dist(aa))
# print(tabu[0])
# print(tabu[1])
# print(tabu[2])
# print(tabu[3])
