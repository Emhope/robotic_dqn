import rospy
from std_msgs.msg import Float32MultiArray

rospy.init_node('cyclo')
pub = rospy.Publisher('/goal_points', Float32MultiArray, queue_size=5)
rate = rospy.Rate(1/30)
f = True
while not rospy.is_shutdown():
    if f:
        m = Float32MultiArray()
        m.data = [1.0, 0.0, 100.0]
        pub.publish(m)
        f = not f
    else:
        m = Float32MultiArray()
        m.data = [6.0, 0.0, 100.0]
        pub.publish(m)
        f = not f
    rate.sleep()
    