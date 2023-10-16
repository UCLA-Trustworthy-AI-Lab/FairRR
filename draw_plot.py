import numpy as np
import matplotlib.pyplot as plt

x= np.arange(1001)/1000
y=0.2 * x +0.4
y1= y+0.02
y0 = y-0.02
y1[y1>1] = 1
y0[y0<0] = 0

plt.subplot(131)

plt.plot(x,y,color='red')
plt.plot(x,y1,'--',color = 'blue')
plt.plot(x,y0,'--',color = 'blue')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0.5,0.5],'--',color = 'green')
plt.plot([0.5,0.5],[0,0.5],'--',color = 'green')
plt.plot([0.4,0.4],[0,0.5],'--',color = 'gold')
plt.plot([0.6,0.6],[0,0.5],'--',color = 'gold')
plt.xticks([0.5],['$t_\delta^\star$'],fontsize = 14)
plt.ylabel('$D_{RC}(t)$',fontsize = 16)


x= np.arange(1001)/1000
y= 0.8 * x + 0.1
y1= y+0.02
y0 = y-0.02
y1[y1>1] = 1
y0[y0<0] = 0


plt.subplot(132)
plt.plot(x,y,color='red')
plt.plot(x,y1,'--',color = 'blue')
plt.plot(x,y0,'--',color = 'blue')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0.5,0.5],'--',color = 'green')
plt.plot([0.5,0.5],[0,0.5],'--',color = 'green')
plt.plot([0.38/0.8,0.38/0.8],[0,0.5],'--',color = 'gold')
plt.plot([0.42/0.8,0.42/0.8],[0,0.5,],'--',color = 'gold')
plt.xticks([0.5],['$t_\delta^\star$'],fontsize = 14)




x= np.arange(501)/1000
y=0.8 * x + 0.05
y1= y+0.02
y0 = y-0.02
y1[y1>1] = 1
y0[y0<0] = 0



xt= np.arange(501)/1000+0.5
yt=0.8 * xt + 0.15
y1t= yt+0.02
y0t = yt-0.02
y1t[y1t>1] = 1
y0t[y0t<0] = 0
plt.subplot(133)

plt.plot(x,y,color='red')
plt.plot(xt,yt,color='red')

plt.plot(x,y1,'--',color = 'blue')
plt.plot(x,y0,'--',color = 'blue')
plt.plot(xt, y1t, '--', color='blue')
plt.plot(xt, y0t, '--', color='blue')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0.5,0.5],'--',color = 'green')
plt.plot([0.5,0.5],[0,1],'--',color = 'green')

plt.xticks([0.5],['$t_\delta^\star$'],fontsize = 14)
plt.show()