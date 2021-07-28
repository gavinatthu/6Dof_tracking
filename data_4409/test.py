import rosbag

def extract(bagfile, outfile):
    f = open(outfile, 'w')
    f.write('# timestamp tx ty tz qx qy qz qw\n')
    with rosbag.Bag(bagfile, 'r') as bag:
        for topic, msg, ts in bag.read_messages(topics = "/amcl_pose"):
            print(msg)
        
if __name__ == '__main__':
    bagfile = 'event_gt_0716_v2_2021-07-16-16-22-28.bag'
    outfile = 'out.txt'
    extract(bagfile, outfile)