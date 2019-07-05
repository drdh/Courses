import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import bk




fig = plt.figure(figsize=(6,20),facecolor='white')
plt.xticks([1,2],[r'$queue\ 1$', r'$queue\ 2$'])
plt.yticks([i for i in range(21)])
#ax = plt.axes(xlim=(0, 10), ylim=(0,10))
#line, = ax.plot([], [], lw=5)
plt.xlim(0,3)
plt.ylim(0,20)
#plt.text(1, 15, r'$Tt$',fontdict={'size': 50, 'color': 'r'})

time=[]
q1=[]
q2=[]
total=[]
totalnum=[]
totaltime=[]
avertime,time,q1,q2,total,totalnum,totaltime=bk.bank(450,time,q1,q2,total,totalnum,totaltime)



N=len(q1)

X=[]
Y=[]

#colors = np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)

for i,j in zip(q1,q2):
    a=[]
    b=[]
    for m in range(i):
        a.append(1)
        b.append(m+1)
    for n in range(j):
        a.append(2)
        b.append(n+1)
    X.append(a)
    Y.append(b)
#plt.plot([0,2],[0,3])

q1_pre=0
q2_pre=0


def init():
     """Clears current frame."""
     line=plt.scatter([],[],s=1500)
     return line,

def animate(i):
     #plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
     #        textcoords='offset points', fontsize=16,
     #        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
#     if i>0 and q2[i] > q2[i-1]:
     
     #plt.text(1, 15, r'$Tt2222$',fontdict={'size': 50, 'color': 'r'})

#         plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
#             textcoords='offset points', fontsize=16,
#             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

     if q1[i/4]>q1[i/4+1]:
         for j in range(q1[i/4]):
             Y[i/4][j]-=0.25
     if q2[i/4]>q2[i/4+1]:
         for j in range(q1[i/4],q1[i/4]+q2[i/4]):
             Y[i/4][j]-=0.25
    # a=fig.plot([1,2],[3,4])
   #  plt.subplot(1,2,2)
     line=plt.scatter(X[i/4],Y[i/4],s=600,c='b',alpha=0.5,marker=(5,2))
     
     if i%4==0 and time[i/4]!=0:
         print "time=%4d    money=%5d    (q1,q2)=(%-3d,%-3d)    num=%3d    totalTime=%6d" % (time[i/4],total[i/4],q1[i/4],q2[i/4],totalnum[i/4],totaltime[i/4])
     elif i%4==0 and time[i/4]==0:
         print "close"
          
    
     return line,

# This call puts the work in motion
# connecting init and animate functions and figure we want to draw
animator = animation.FuncAnimation(fig, animate, init_func=init, frames=4*(N-2), interval=120, blit=True)

plt.show()

