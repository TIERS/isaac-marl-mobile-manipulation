import sys
import rospy
import moveit_commander

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

# print joint values
#joint_values = move_group.get_current_joint_values()
#print("joint values: ", joint_values)

# move to joint values
joint_goal = [
    0.0005240757545913781,
    -0.7851120893497162,
    -0.0001942464402225399,
    -2.3565575075316847,
    0.0008870231278934467,
    1.5711619433297048,
    0.7842869315418955,
]
move_group.go(joint_goal, wait=True)
