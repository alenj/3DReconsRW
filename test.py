

def ploting_code(gps1,gps2,gps3,gps4):

    load data
    gps = [gps1;gps2;gps3; gps4]-gps1
    plt.figure(1)
    plot3(gps(:,1), gps(:,2), gps(:,3), 'or'); hold on
    plot3(XP(1,:), XP(2,:), XP(3,:),'ob'); hold off
    grid on;
    axis('equal')

    x = gps(2,1);
    y = gps(2,2);
    alpha = x/y;

    x2 = x2p(:,1)
    x1 = x1p(:,1)

    y1 = x1p(:,2)
    y2 = x2p(:,2)

    cost_fun

    return


def  cost_fun(R1, R2, R3, R4, x1,x2,y1,y2,alpha):

    c = x2-R1*x1 - R2*y1 -alpha(y2-R3*x1-R4*y1)

    return  c

